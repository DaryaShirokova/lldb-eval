/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/fuzzer/expr_gen.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "lldb-eval/defines.h"
#include "tools/fuzzer/ast.h"

namespace fuzzer {

class Weights {
 public:
  using ExprWeightsArray = std::array<float, NUM_GEN_EXPR_KINDS>;
  using TypeWeightsArray = std::array<float, NUM_GEN_TYPE_KINDS>;

  ExprWeightsArray& expr_weights() { return expr_weights_; }
  const ExprWeightsArray& expr_weights() const { return expr_weights_; }

  TypeWeightsArray& type_weights() { return type_weights_; }
  const TypeWeightsArray& type_weights() const { return type_weights_; }

  float& operator[](ExprKind kind) { return expr_weights_[(size_t)kind]; }
  float& operator[](TypeKind kind) { return type_weights_[(size_t)kind]; }

  const float& operator[](ExprKind kind) const {
    return expr_weights_[(size_t)kind];
  }
  const float& operator[](TypeKind kind) const {
    return type_weights_[(size_t)kind];
  }

 private:
  std::array<float, NUM_GEN_EXPR_KINDS> expr_weights_;
  std::array<float, NUM_GEN_TYPE_KINDS> type_weights_;
};

Expr fallback_expr(const QualifiedType& type) {
  struct FallbackVisitor {
    Expr operator()(const PointerType&) { return IntegerConstant(0); }
    Expr operator()(const ScalarType&) { return IntegerConstant(0xFA17); }
    Expr operator()(const TaggedType&) { return VariableExpr("ts"); }
  };

  return std::visit(FallbackVisitor(), type.type());
}

enum class AllowedTypeKinds {
  // Any integer type or any time implicitly treated as an integer type
  // (e.g. `bool`) is allowed.
  AnyInt,

  // Any type that can be implicitly treated as an integer or floating point
  // type (e.g. `int`, `bool`, `float`) is allowed.
  AnyFloat,

  // Any pointer type.
  AnyPointer,

  // Any type that can be used in a boolean context (i.e. the condition in an
  // `if` or `while` statement, and all operands to the `&&`, `||` and `!`
  // operators) is allowed.
  AnyForBoolContext,

  // Any type (useful for e.g. `(void) expr`).
  AnyType,
};

using AllowedExprKinds = std::bitset<NUM_GEN_EXPR_KINDS>;

using TypeToGen =
    std::variant<AllowedTypeKinds, std::reference_wrapper<const QualifiedType>,
                 std::reference_wrapper<const ReferenceType>>;

class TypeConstraints {
 public:
  bool must_be_lvalue() const { return must_be_lvalue_; }
  const TypeToGen& type() const { return type_; }

 private:
  bool must_be_lvalue_ = false;

  TypeToGen type_;
};

class AllowedExprKindsVisitor {
 public:
  AllowedExprKinds operator()(AllowedTypeKinds kinds) {
    if (kinds == AllowedTypeKinds::AnyType) {
      return AllowedExprKinds(~0ull);
    }

    AllowedExprKinds retval = 0;

    // `0` is the null pointer constant, hence we can always use `0` as an
    // integer constant.
    retval[(size_t)ExprKind::IntegerConstant] = true;

    // We can always generate ternary expressions (e.g. a ternary exception
    // with the same left hand side and right hand side).
    retval[(size_t)ExprKind::TernaryExpr] = true;

    // We can always generate cast expressions (e.g. a no-op cast).
    retval[(size_t)ExprKind::CastExpr] = true;

    // We can always generate `(expr + 0)` for a given expression.
    retval[(size_t)ExprKind::BinaryExpr] = true;

    retval[(size_t)ExprKind::VariableExpr] = true;

    if (kinds != AllowedTypeKinds::AnyPointer) {
      // We can only use `!` for pointers if we're concerned about a boolean
      // context.
      retval[(size_t)ExprKind::UnaryExpr] = true;
      // `false` is not a null pointer constant.
      retval[(size_t)ExprKind::BooleanConstant] = true;
    }

    if (kinds == AllowedTypeKinds::AnyFloat ||
        kinds == AllowedTypeKinds::AnyForBoolContext) {
      retval[(size_t)ExprKind::DoubleConstant] = true;
    }

    // TODO(alextasos): Deal with the following cases:
    // retval[(size_t)ExprKind::AddressOf] = true;
    // retval[(size_t)ExprKind::MemberOf] = true;
    // retval[(size_t)ExprKind::MemberOfPtr] = true;
    // retval[(size_t)ExprKind::ArrayIndex] = true;

    return retval;
  }

  AllowedExprKinds operator()(const ReferenceType& type) {
    auto retval = (*this)(type.type());
    if (!type.can_reference_rvalue()) {
      retval[(size_t)ExprKind::IntegerConstant] = false;
      retval[(size_t)ExprKind::BooleanConstant] = false;
      retval[(size_t)ExprKind::DoubleConstant] = false;
      retval[(size_t)ExprKind::UnaryExpr] = false;
      retval[(size_t)ExprKind::TernaryExpr] = false;
      retval[(size_t)ExprKind::CastExpr] = false;
    }

    return retval;
  }

  AllowedExprKinds operator()(const QualifiedType& type) {
    return std::visit(*this, type.type());
  }

  AllowedExprKinds operator()(const ScalarType& type) {
    AllowedExprKinds retval = ~0ull;
    if (type == ScalarType::Void) {
      return retval;
    }
    retval[(size_t)ExprKind::AddressOf] = false;

    if (type == ScalarType::Float || type == ScalarType::Double ||
        type == ScalarType::LongDouble) {
      return retval;
    }
    retval[(size_t)ExprKind::DoubleConstant] = false;

    return retval;
  }

  AllowedExprKinds operator()(const TaggedType&) {
    AllowedExprKinds retval = 0;
    retval[(size_t)ExprKind::VariableExpr] = true;
    retval[(size_t)ExprKind::CastExpr] = false;

    return retval;
  }

  AllowedExprKinds operator()(const PointerType& type) {
    AllowedExprKinds retval = 0;
    retval[(size_t)ExprKind::IntegerConstant] = true;

    if (std::holds_alternative<PointerType>(type.type().type())) {
      return retval;
    }

    retval[(size_t)ExprKind::AddressOf] = true;
    retval[(size_t)ExprKind::VariableExpr] = true;

    return retval;
  }
};

AllowedExprKinds allowed_expr_kinds(const TypeConstraints constraints) {
  auto retval = std::visit(AllowedExprKindsVisitor(), constraints.type());
  if (constraints.must_be_lvalue()) {
    // TODO(alextasos): Duplicate code below
    retval[(size_t)ExprKind::IntegerConstant] = false;
    retval[(size_t)ExprKind::DoubleConstant] = false;
    retval[(size_t)ExprKind::UnaryExpr] = false;
    retval[(size_t)ExprKind::TernaryExpr] = false;
    retval[(size_t)ExprKind::CastExpr] = false;
  }

  return retval;
}

int expr_precedence(const Expr& e) {
  return std::visit([](const auto& e) { return e.precedence(); }, e);
}

BooleanConstant ExprGenerator::gen_boolean_constant() {
  return BooleanConstant(rng_->gen_boolean());
}

IntegerConstant ExprGenerator::gen_integer_constant(const TypeConstraints&) {
  auto value = rng_->gen_u64(cfg_.int_const_min, cfg_.int_const_max);

  return IntegerConstant(value);
}

DoubleConstant ExprGenerator::gen_double_constant() {
  auto value =
      rng_->gen_double(cfg_.double_constant_min, cfg_.double_constant_max);

  return DoubleConstant(value);
}

VariableExpr ExprGenerator::gen_variable_expr(const TypeConstraints&) {
  return VariableExpr(VAR);
}

BinaryExpr ExprGenerator::gen_binary_expr(const Weights& weights,
                                          const TypeConstraints& constraints) {
  auto op = rng_->gen_bin_op(cfg_.bin_op_mask);

  auto lhs = gen_with_weights(weights, constraints);
  auto rhs = gen_with_weights(weights, constraints);

  // Rules for parenthesising the left hand side:
  // 1. If the left hand side has a strictly lower precedence than ours,
  //    then we will have to emit parens.
  //    Example: We emit `(3 + 4) * 5` instead of `3 + 4 * 5`.
  // 2. If the left hand side has the same precedence as we do, then we
  //    don't have to emit any parens. This is because all lldb-eval
  //    binary operators have left-to-right associativity.
  //    Example: We do not have to emit `(3 - 4) + 5`, `3 - 4 + 5` will also
  //    do.
  auto lhs_precedence = expr_precedence(lhs);
  if (lhs_precedence > bin_op_precedence(op)) {
    lhs = ParenthesizedExpr(std::move(lhs));
  }

  // Rules for parenthesising the right hand side:
  // 1. If the right hand side has a strictly lower precedence than ours,
  //    then we will have to emit parens.
  //    Example: We emit `5 * (3 + 4)` instead of `5 * 3 + 4`.
  // 2. If the right hand side has the same precedence as we do, then we
  //    should emit parens for good measure. This is because all lldb-eval
  //    binary operators have left-to-right associativity and we do not
  //    want to violate this with respect to the generated AST.
  //    Example: We emit `3 - (4 + 5)` instead of `3 - 4 + 5`. We also
  //    emit `3 + (4 + 5)` instead of `3 + 4 + 5`, even though both
  //    expressions are equivalent.
  auto rhs_precedence = expr_precedence(rhs);
  if (rhs_precedence >= bin_op_precedence(op)) {
    rhs = ParenthesizedExpr(std::move(rhs));
  }

  return BinaryExpr(std::move(lhs), op, std::move(rhs));
}

UnaryExpr ExprGenerator::gen_unary_expr(const Weights& weights,
                                        const TypeConstraints& constraints) {
  auto expr = gen_with_weights(weights, constraints);
  auto op = (UnOp)rng_->gen_un_op(cfg_.un_op_mask);

  if (expr_precedence(expr) > UnaryExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return UnaryExpr(op, std::move(expr));
}

TernaryExpr ExprGenerator::gen_ternary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto cond = gen_with_weights(weights, constraints);
  auto lhs = gen_with_weights(weights, constraints);
  auto rhs = gen_with_weights(weights, constraints);

  if (expr_precedence(cond) == TernaryExpr::PRECEDENCE) {
    cond = ParenthesizedExpr(std::move(cond));
  }

  return TernaryExpr(std::move(cond), std::move(lhs), std::move(rhs));
}

CastExpr ExprGenerator::gen_cast_expr(const Weights& weights,
                                      const TypeConstraints& constraints) {
  auto type = gen_type(weights, constraints);
  auto expr = gen_with_weights(weights, constraints);

  if (expr_precedence(expr) > CastExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return CastExpr(std::move(type), std::move(expr));
}

AddressOf ExprGenerator::gen_address_of_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto expr = gen_with_weights(weights, constraints);
  if (expr_precedence(expr) > AddressOf::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return AddressOf(std::move(expr));
}

MemberOf ExprGenerator::gen_member_of_expr(const Weights& weights,
                                           const TypeConstraints& constraints) {
  auto expr = gen_with_weights(weights, constraints);
  if (expr_precedence(expr) > MemberOf::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return MemberOf(std::move(expr), "f1");
}

MemberOfPtr ExprGenerator::gen_member_of_ptr_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto expr = gen_with_weights(weights, constraints);
  if (expr_precedence(expr) > MemberOfPtr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return MemberOfPtr(std::move(expr), "f1");
}

ArrayIndex ExprGenerator::gen_array_index_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto expr = gen_with_weights(weights, constraints);
  auto idx = gen_with_weights(weights, constraints);
  if (expr_precedence(expr) > ArrayIndex::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return ArrayIndex(std::move(expr), std::move(idx));
}

Expr ExprGenerator::gen_with_weights(const Weights& weights,
                                     const TypeConstraints& constraints) {
  Weights new_weights = weights;

  auto kind = rng_->gen_expr_kind(new_weights);
  auto idx = (size_t)kind;
  new_weights[kind] *= cfg_.expr_kind_weights[idx].dampening_factor;

  // Dummy value for initialization
  Expr expr(IntegerConstant(0));
  switch (kind) {
    case ExprKind::IntegerConstant:
      expr = gen_integer_constant(constraints);
      break;

    case ExprKind::DoubleConstant:
      expr = gen_double_constant();
      break;

    case ExprKind::VariableExpr:
      expr = gen_variable_expr(constraints);
      break;

    case ExprKind::BinaryExpr:
      expr = gen_binary_expr(new_weights, constraints);
      break;

    case ExprKind::UnaryExpr:
      expr = gen_unary_expr(new_weights, constraints);
      break;

    case ExprKind::TernaryExpr:
      expr = gen_ternary_expr(new_weights, constraints);
      break;

    case ExprKind::BooleanConstant:
      expr = gen_boolean_constant();
      break;

    case ExprKind::CastExpr:
      expr = gen_cast_expr(new_weights, constraints);
      break;

    case ExprKind::AddressOf:
      expr = gen_address_of_expr(new_weights, constraints);
      break;

    case ExprKind::MemberOf:
      expr = gen_member_of_expr(new_weights, constraints);
      break;

    case ExprKind::MemberOfPtr:
      expr = gen_member_of_ptr_expr(new_weights, constraints);
      break;

    case ExprKind::ArrayIndex:
      expr = gen_array_index_expr(new_weights, constraints);
      break;

    default:
      lldb_eval_unreachable("Unhandled expression generation case");
  }

  return maybe_parenthesized(std::move(expr));
}

Expr ExprGenerator::maybe_parenthesized(Expr expr) {
  if (rng_->gen_parenthesize(cfg_.parenthesize_prob)) {
    return ParenthesizedExpr(std::move(expr));
  }

  return expr;
}

Type ExprGenerator::gen_type(const Weights& weights,
                             const TypeConstraints& constraints) {
  Weights new_weights = weights;
  auto choice = rng_->gen_type_kind(new_weights);
  auto idx = (size_t)choice;

  auto& new_type_weights = new_weights.type_weights();
  new_type_weights[idx] *= cfg_.type_kind_weights[idx].dampening_factor;

  QualifiableType type;
  switch (choice) {
    case TypeKind::ScalarType:
      type = gen_scalar_type(constraints);
      break;

    case TypeKind::TaggedType:
      type = gen_tagged_type(constraints);
      break;

    case TypeKind::PointerType:
      type = gen_pointer_type(new_weights, constraints);
      break;

    case TypeKind::ReferenceType: {
      auto qualified_type = gen_qualified_type(new_weights, constraints);
      return ReferenceType(std::move(qualified_type));
    }
  }

  auto qualifiers = gen_cv_qualifiers();
  return QualifiedType(std::move(type), qualifiers);
}

PointerType ExprGenerator::gen_pointer_type(
    const Weights& weights, const TypeConstraints& constraints) {
  auto type = gen_qualified_type(weights, constraints);

  return PointerType(std::move(type));
}

QualifiedType ExprGenerator::gen_qualified_type(
    const Weights& weights, const TypeConstraints& constraints) {
  Weights new_weights = weights;
  auto& new_type_weights = new_weights.type_weights();
  // Reference types are not qualified types, hence don't generate any
  new_type_weights[(size_t)TypeKind::ReferenceType] = 0;

  auto choice = rng_->gen_type_kind(new_weights);
  auto idx = (size_t)choice;

  new_type_weights[idx] *= cfg_.type_kind_weights[idx].dampening_factor;

  QualifiableType type;
  switch (choice) {
    case TypeKind::ScalarType:
      type = gen_scalar_type(constraints);
      break;

    case TypeKind::TaggedType:
      type = gen_tagged_type(constraints);
      break;

    case TypeKind::PointerType:
      type = gen_pointer_type(weights, constraints);
      break;

    default:
      assert(false && "Unreachable");
      return QualifiedType(ScalarType::Void);
  }
  auto qualifiers = gen_cv_qualifiers();

  return QualifiedType(std::move(type), qualifiers);
}

TaggedType ExprGenerator::gen_tagged_type(const TypeConstraints&) {
  return TaggedType("TestStruct");
}

ScalarType ExprGenerator::gen_scalar_type(const TypeConstraints&) {
  return rng_->gen_scalar_type();
}

CvQualifiers ExprGenerator::gen_cv_qualifiers() {
  return rng_->gen_cv_qualifiers(cfg_.const_prob, cfg_.volatile_prob);
}

Expr ExprGenerator::generate() {
  Weights weights;
  TypeConstraints constraints;

  auto& expr_weights = weights.expr_weights();
  for (size_t i = 0; i < expr_weights.size(); i++) {
    expr_weights[i] = cfg_.expr_kind_weights[i].initial_weight;
  }

  auto& type_weights = weights.type_weights();
  for (size_t i = 0; i < type_weights.size(); i++) {
    type_weights[i] = cfg_.type_kind_weights[i].initial_weight;
  }

  return gen_with_weights(weights, constraints);
}

template <size_t N, typename Rng>
size_t pick_nth_set_bit(std::bitset<N> mask, Rng& rng) {
  // At least one bit needs to be set
  assert(mask.any() && "Mask must not be empty");

  std::uniform_int_distribution<size_t> distr(1, mask.count());
  size_t choice = distr(rng);

  size_t running_ones = 0;
  for (size_t i = 0; i < mask.size(); i++) {
    if (mask[i]) {
      running_ones++;
    }

    if (running_ones == choice) {
      return i;
    }
  }

  // `choice` lies in the range `[1, mask.count()]`, `running_ones` will
  // always lie in the range `[0, mask.count()]` and is incremented at most
  // once per loop iteration. The only way for this assertion to fire is for
  // `mask` to be empty (which we have asserted beforehand).
  lldb_eval_unreachable("Mask has no bits set");
}

template <size_t N, typename Rng, typename RealType>
size_t weighted_pick(const std::array<RealType, N>& array, Rng& rng) {
  static_assert(N != 0, "Array must have at least 1 element");
  static_assert(std::is_floating_point_v<RealType>,
                "Must be a floating point type");

  RealType sum = 0;
  for (const auto& e : array) {
    sum += e;
  }

  std::uniform_real_distribution<RealType> distr(0, sum);
  RealType choice = distr(rng);

  RealType running_sum = 0;
  for (size_t i = 0; i < array.size(); i++) {
    running_sum += array[i];
    if (choice < running_sum) {
      return i;
    }
  }

  // Just in case we get here due to e.g. floating point inaccuracies, etc.
  return array.size() - 1;
}

BinOp DefaultGeneratorRng::gen_bin_op(BinOpMask mask) {
  return (BinOp)pick_nth_set_bit(mask, rng_);
}

UnOp DefaultGeneratorRng::gen_un_op(UnOpMask mask) {
  return (UnOp)pick_nth_set_bit(mask, rng_);
}

uint64_t DefaultGeneratorRng::gen_u64(uint64_t min, uint64_t max) {
  std::uniform_int_distribution<uint64_t> distr(min, max);
  return distr(rng_);
}

double DefaultGeneratorRng::gen_double(double min, double max) {
  std::uniform_real_distribution<double> distr(min, max);
  return distr(rng_);
}

CvQualifiers DefaultGeneratorRng::gen_cv_qualifiers(float const_prob,
                                                    float volatile_prob) {
  std::bernoulli_distribution const_distr(const_prob);
  std::bernoulli_distribution volatile_distr(volatile_prob);

  CvQualifiers retval = 0;
  if (const_distr(rng_)) {
    retval.set((size_t)CvQualifier::Const);
  }
  if (volatile_distr(rng_)) {
    retval.set((size_t)CvQualifier::Volatile);
  }

  return retval;
}

bool DefaultGeneratorRng::gen_parenthesize(float probability) {
  std::bernoulli_distribution distr(probability);
  return distr(rng_);
}

bool DefaultGeneratorRng::gen_boolean() {
  std::bernoulli_distribution distr;
  return distr(rng_);
}

ExprKind DefaultGeneratorRng::gen_expr_kind(const Weights& weights) {
  return (ExprKind)weighted_pick(weights.expr_weights(), rng_);
}

TypeKind DefaultGeneratorRng::gen_type_kind(const Weights& weights) {
  return (TypeKind)weighted_pick(weights.type_weights(), rng_);
}

ScalarType DefaultGeneratorRng::gen_scalar_type() {
  std::uniform_int_distribution<int> distr((int)ScalarType::EnumMin,
                                           (int)ScalarType::EnumMax);
  return (ScalarType)distr(rng_);
}

}  // namespace fuzzer
