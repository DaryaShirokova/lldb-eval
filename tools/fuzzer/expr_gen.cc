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
#include <optional>
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

enum class AllowedTypeKind {
  // Any integer type or any time implicitly treated as an integer type
  // (e.g. `bool`) is allowed.
  AnyInt,

  // Any type that can be implicitly treated as an integer or floating point
  // type (e.g. `int`, `bool`, `float`) is allowed.
  AnyFloat,

  // Any type that can be used in a boolean context (i.e. the condition in an
  // `if` or `while` statement, and all operands to the `&&`, `||` and `!`
  // operators) is allowed.
  AnyForBoolContext,

  // Any type (useful for e.g. `(void) expr`).
  AnyType,
};

using TypeToGen = std::variant<AllowedTypeKind, Type>;

class TypeConstraints {
 public:
  const TypeToGen& gen_type() const { return type_; }
  bool must_be_lvalue() const { return must_be_lvalue_; }

  bool is_allowed_type_kind() const {
    return std::holds_alternative<AllowedTypeKind>(type_);
  }
  const AllowedTypeKind* as_allowed_type_kind() const {
    return std::get_if<AllowedTypeKind>(&type_);
  }

  bool is_type() const { return std::holds_alternative<Type>(type_); }
  const Type* as_type() const { return std::get_if<Type>(&type_); }

  bool is_pointer_type() const {
    return std::holds_alternative<Type>(type_);
    const auto* ptr = std::get_if<Type>(&type_);
    if (ptr == nullptr) {
      return false;
    }
    return std::holds_alternative<PointerType>(*ptr);
  }
  const PointerType* as_pointer_type() const {
    const auto* ptr = std::get_if<Type>(&type_);
    if (ptr == nullptr) {
      return nullptr;
    }

    return std::get_if<PointerType>(ptr);
  }

  bool is_scalar_type() const {
    return std::holds_alternative<Type>(type_);
    const auto* ptr = std::get_if<Type>(&type_);
    if (ptr == nullptr) {
      return false;
    }
    return std::holds_alternative<ScalarType>(*ptr);
  }
  const ScalarType* as_scalar_type() const {
    const auto* ptr = std::get_if<Type>(&type_);
    if (ptr == nullptr) {
      return nullptr;
    }

    return std::get_if<ScalarType>(ptr);
  }

  bool is_tagged_type() const {
    return std::holds_alternative<Type>(type_);
    const auto* ptr = std::get_if<Type>(&type_);
    if (ptr == nullptr) {
      return false;
    }
    return std::holds_alternative<TaggedType>(*ptr);
  }
  const TaggedType* as_tagged_type() const {
    const auto* ptr = std::get_if<Type>(&type_);
    if (ptr == nullptr) {
      return nullptr;
    }

    return std::get_if<TaggedType>(ptr);
  }

  explicit TypeConstraints(TypeToGen type, bool must_be_lvalue = false)
      : type_(std::move(type)), must_be_lvalue_(must_be_lvalue) {}

 private:
  TypeToGen type_;
  bool must_be_lvalue_ = false;
};

/*
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
*/

template <typename T, typename CrtpClass>
class ConstraintsBaseVisitor {
 public:
  T operator()(const AllowedTypeKind&) {
    lldb_eval_unreachable("Generation not implemented for type kinds.");
  }

  T operator()(const PointerType&) {
    lldb_eval_unreachable("Generation not implemented for pointer types.");
  }

  T operator()(const TaggedType&) {
    lldb_eval_unreachable("Generation not implemented for tagged types.");
  }

  T operator()(const ScalarType&) {
    lldb_eval_unreachable("Generation not implemented for scalar types.");
  }

  T operator()(const Type& type) {
    return std::visit(static_cast<CrtpClass&>(*this), type);
  }

  T operator()(const QualifiedType& type) {
    return std::visit(static_cast<CrtpClass&>(*this), type.type());
  }
};

int expr_precedence(const Expr& e) {
  return std::visit([](const auto& e) { return e.precedence(); }, e);
}

std::optional<Expr> ExprGenerator::gen_boolean_constant(
    const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  if (constraints.is_allowed_type_kind() || constraints.is_scalar_type()) {
    return BooleanConstant(rng_->gen_boolean());
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_integer_constant(
    const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  if (constraints.is_pointer_type()) {
    return IntegerConstant(0);
  }

  if (constraints.is_allowed_type_kind() || constraints.is_scalar_type()) {
    return rng_->gen_integer_constant(cfg_.int_const_min, cfg_.int_const_max);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_double_constant(
    const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  const auto* as_kind = constraints.as_allowed_type_kind();
  if (as_kind != nullptr && *as_kind != AllowedTypeKind::AnyInt) {
    return rng_->gen_double_constant(cfg_.double_constant_min,
                                     cfg_.double_constant_max);
  }

  if (constraints.is_scalar_type()) {
    return rng_->gen_double_constant(cfg_.double_constant_min,
                                     cfg_.double_constant_max);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_variable_expr(
    const TypeConstraints& constraints) {
  class Visitor : public ConstraintsBaseVisitor<VariableExpr, Visitor> {
   public:
    Visitor(ExprGenerator& gen) : gen_(gen) {}

    using ConstraintsBaseVisitor::operator();

    VariableExpr operator()(const AllowedTypeKind&) {
      // TODO: Generate more kinds of variables
      return VariableExpr("x");
    }

    VariableExpr operator()(const Type& type) {
      const auto& symtab = gen_.cfg_.symbol_table;
      auto res = symtab.find(type);
      assert(res != symtab.end() && "Could not find type in symbol table");
      assert(!res->second.empty() && "Could not find variables with this type");

      return VariableExpr(res->second[0]);
    }

   private:
    ExprGenerator& gen_;
  };

  return std::visit(Visitor(*this), *constraints.as_type());
}

std::optional<Expr> ExprGenerator::gen_binary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto op = rng_->gen_bin_op(cfg_.bin_op_mask);

  auto maybe_lhs = gen_with_weights(weights, constraints);
  if (!maybe_lhs.has_value()) {
    return {};
  }
  auto& lhs = maybe_lhs.value();

  auto maybe_rhs = gen_with_weights(weights, constraints);
  if (!maybe_rhs.has_value()) {
    return {};
  }
  auto& rhs = maybe_rhs.value();

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

std::optional<Expr> ExprGenerator::gen_unary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  static constexpr UnOpMask int_only_mask = 1ull << (size_t)UnOp::BitNot;
  auto mask = cfg_.un_op_mask;

  const auto* as_kind = constraints.as_allowed_type_kind();
  if (as_kind != nullptr && *as_kind != AllowedTypeKind::AnyInt) {
    mask &= ~int_only_mask;
  }

  const auto* as_scalar = constraints.as_scalar_type();
  if (as_scalar != nullptr && *as_scalar != ScalarType::Float &&
      *as_scalar != ScalarType::Double &&
      *as_scalar != ScalarType::LongDouble) {
    mask &= ~int_only_mask;
  }

  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto& expr = maybe_expr.value();
  auto op = (UnOp)rng_->gen_un_op(mask);

  if (expr_precedence(expr) > UnaryExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return UnaryExpr(op, std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_ternary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_cond = gen_with_weights(
      weights, TypeConstraints(AllowedTypeKind::AnyForBoolContext));
  if (!maybe_cond.has_value()) {
    return {};
  }
  auto& cond = maybe_cond.value();

  auto maybe_lhs = gen_with_weights(weights, constraints);
  if (!maybe_lhs.has_value()) {
    return {};
  }
  auto& lhs = maybe_lhs.value();

  auto maybe_rhs = gen_with_weights(weights, constraints);
  if (!maybe_rhs.has_value()) {
    return {};
  }
  auto& rhs = maybe_lhs.value();

  if (expr_precedence(cond) == TernaryExpr::PRECEDENCE) {
    cond = ParenthesizedExpr(std::move(cond));
  }

  return TernaryExpr(std::move(cond), std::move(lhs), std::move(rhs));
}

std::optional<Expr> ExprGenerator::gen_cast_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  TypeConstraints new_constraints(AllowedTypeKind::AnyType);

  auto maybe_type = gen_type(weights, constraints);
  if (!maybe_type.has_value()) {
    return {};
  }
  auto& type = maybe_type.value();

  auto maybe_expr = gen_with_weights(weights, new_constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto& expr = maybe_expr.value();

  if (expr_precedence(expr) > CastExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return CastExpr(std::move(type), std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_address_of_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto& expr = maybe_expr.value();

  if (expr_precedence(expr) > AddressOf::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return AddressOf(std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_member_of_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto& expr = maybe_expr.value();

  if (expr_precedence(expr) > MemberOf::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return MemberOf(std::move(expr), "f1");
}

std::optional<Expr> ExprGenerator::gen_member_of_ptr_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto& expr = maybe_expr.value();

  if (expr_precedence(expr) > MemberOfPtr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return MemberOfPtr(std::move(expr), "f1");
}

std::optional<Expr> ExprGenerator::gen_array_index_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  TypeConstraints idx_constraints(AllowedTypeKind::AnyInt);

  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto& expr = maybe_expr.value();

  auto maybe_idx = gen_with_weights(weights, idx_constraints);
  if (!maybe_idx.has_value()) {
    return {};
  }
  auto& idx = maybe_idx.value();

  if (expr_precedence(expr) > ArrayIndex::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return ArrayIndex(std::move(expr), std::move(idx));
}

std::optional<Expr> ExprGenerator::gen_with_weights(
    const Weights& weights, const TypeConstraints& constraints) {
  Weights new_weights = weights;

  ExprKindMask mask = ~0llu;

  while (mask.any()) {
    auto kind = rng_->gen_expr_kind(new_weights, mask);
    auto idx = (size_t)kind;
    auto old_weight = new_weights[kind];
    new_weights[kind] *= cfg_.expr_kind_weights[idx].dampening_factor;

    std::optional<Expr> maybe_expr;
    switch (kind) {
      case ExprKind::IntegerConstant:
        maybe_expr = gen_integer_constant(constraints);
        break;

      case ExprKind::DoubleConstant:
        maybe_expr = gen_double_constant(constraints);
        break;

      case ExprKind::VariableExpr:
        maybe_expr = gen_variable_expr(constraints);
        break;

      case ExprKind::BinaryExpr:
        maybe_expr = gen_binary_expr(new_weights, constraints);
        break;

      case ExprKind::UnaryExpr:
        maybe_expr = gen_unary_expr(new_weights, constraints);
        break;

      case ExprKind::TernaryExpr:
        maybe_expr = gen_ternary_expr(new_weights, constraints);
        break;

      case ExprKind::BooleanConstant:
        maybe_expr = gen_boolean_constant(constraints);
        break;

      case ExprKind::CastExpr:
        maybe_expr = gen_cast_expr(new_weights, constraints);
        break;

      case ExprKind::AddressOf:
        maybe_expr = gen_address_of_expr(new_weights, constraints);
        break;

      case ExprKind::MemberOf:
        maybe_expr = gen_member_of_expr(new_weights, constraints);
        break;

      case ExprKind::MemberOfPtr:
        maybe_expr = gen_member_of_ptr_expr(new_weights, constraints);
        break;

      case ExprKind::ArrayIndex:
        maybe_expr = gen_array_index_expr(new_weights, constraints);
        break;

      default:
        lldb_eval_unreachable("Unhandled expression generation case");
    }

    if (!maybe_expr.has_value()) {
      new_weights[kind] = old_weight;
      mask[idx] = false;
    }
    auto& expr = maybe_expr.value();

    return maybe_parenthesized(std::move(expr));
  }

  return {};
}

Expr ExprGenerator::maybe_parenthesized(Expr expr) {
  if (rng_->gen_parenthesize(cfg_.parenthesize_prob)) {
    return ParenthesizedExpr(std::move(expr));
  }

  return expr;
}

std::optional<Type> ExprGenerator::gen_type(
    const Weights& weights, const TypeConstraints& constraints) {
  Weights new_weights = weights;

  TypeKindMask mask = ~0llu;

  while (mask.any()) {
    auto choice = rng_->gen_type_kind(new_weights, mask);
    auto idx = (size_t)choice;

    auto& new_type_weights = new_weights.type_weights();
    auto old_weight = new_type_weights[idx];
    new_type_weights[idx] *= cfg_.type_kind_weights[idx].dampening_factor;

    std::optional<Type> maybe_type;
    switch (choice) {
      case TypeKind::ScalarType:
        maybe_type = gen_scalar_type(constraints);
        break;

      case TypeKind::TaggedType:
        maybe_type = gen_tagged_type(constraints);
        break;

      case TypeKind::PointerType:
        maybe_type = gen_pointer_type(new_weights, constraints);
        break;
    }

    if (maybe_type.has_value()) {
      return maybe_type;
    }

    new_type_weights[idx] = old_weight;
    mask[idx] = false;
  }

  return {};
}

std::optional<QualifiedType> ExprGenerator::gen_qualified_type(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_type = gen_type(weights, constraints);
  if (!maybe_type.has_value()) {
    return {};
  }
  auto& type = maybe_type.value();
  auto qualifiers = gen_cv_qualifiers();

  return QualifiedType(std::move(type), qualifiers);
}

std::optional<Type> ExprGenerator::gen_pointer_type(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_type = gen_qualified_type(weights, constraints);
  if (!maybe_type.has_value()) {
    return {};
  }
  auto& type = maybe_type.value();

  return PointerType(std::move(type));
}

std::optional<Type> ExprGenerator::gen_tagged_type(const TypeConstraints&) {
  return TaggedType("TestStruct");
}

std::optional<Type> ExprGenerator::gen_scalar_type(const TypeConstraints&) {
  return rng_->gen_scalar_type();
}

CvQualifiers ExprGenerator::gen_cv_qualifiers() {
  return rng_->gen_cv_qualifiers(cfg_.const_prob, cfg_.volatile_prob);
}

std::optional<Expr> ExprGenerator::generate() {
  Weights weights;

  auto& expr_weights = weights.expr_weights();
  for (size_t i = 0; i < expr_weights.size(); i++) {
    expr_weights[i] = cfg_.expr_kind_weights[i].initial_weight;
  }

  auto& type_weights = weights.type_weights();
  for (size_t i = 0; i < type_weights.size(); i++) {
    type_weights[i] = cfg_.type_kind_weights[i].initial_weight;
  }

  return gen_with_weights(weights, TypeConstraints(AllowedTypeKind::AnyType));
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
size_t weighted_pick(const std::array<RealType, N>& array,
                     const std::bitset<N>& mask, Rng& rng) {
  static_assert(N != 0, "Array must have at least 1 element");
  static_assert(std::is_floating_point_v<RealType>,
                "Must be a floating point type");

  RealType sum = 0;
  for (size_t i = 0; i < array.size(); i++) {
    sum += mask[i] ? array[i] : 0;
  }

  std::uniform_real_distribution<RealType> distr(0, sum);
  RealType choice = distr(rng);

  RealType running_sum = 0;
  for (size_t i = 0; i < array.size(); i++) {
    running_sum += mask[i] ? array[i] : 0;
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

IntegerConstant DefaultGeneratorRng::gen_integer_constant(uint64_t min,
                                                          uint64_t max) {
  using Base = IntegerConstant::Base;
  using Length = IntegerConstant::Length;
  using Signedness = IntegerConstant::Signedness;

  std::uniform_int_distribution<uint64_t> distr(min, max);
  auto value = distr(rng_);

  std::uniform_int_distribution<int> base_distr((int)Base::EnumFirst,
                                                (int)Base::EnumLast);
  auto base = (Base)base_distr(rng_);

  std::uniform_int_distribution<int> length_distr((int)Length::EnumFirst,
                                                  (int)Length::EnumLast);
  auto length = (Length)base_distr(rng_);

  std::uniform_int_distribution<int> sign_distr((int)Signedness::EnumFirst,
                                                (int)Signedness::EnumLast);
  auto signedness = (Signedness)base_distr(rng_);

  return IntegerConstant(value, base, length, signedness);
}

DoubleConstant DefaultGeneratorRng::gen_double_constant(double min,
                                                        double max) {
  using Format = DoubleConstant::Format;
  using Length = DoubleConstant::Length;

  std::uniform_real_distribution<double> distr(min, max);
  auto value = distr(rng_);

  std::uniform_int_distribution<int> format_distr((int)Format::EnumFirst,
                                                  (int)Format::EnumLast);
  auto format = (Format)format_distr(rng_);

  std::uniform_int_distribution<int> length_distr((int)Length::EnumFirst,
                                                  (int)Length::EnumLast);
  auto length = (Length)length_distr(rng_);

  return DoubleConstant(value, format, length);
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

ExprKind DefaultGeneratorRng::gen_expr_kind(const Weights& weights,
                                            const ExprKindMask& mask) {
  return (ExprKind)weighted_pick(weights.expr_weights(), mask, rng_);
}

TypeKind DefaultGeneratorRng::gen_type_kind(const Weights& weights,
                                            const TypeKindMask& mask) {
  return (TypeKind)weighted_pick(weights.type_weights(), mask, rng_);
}

ScalarType DefaultGeneratorRng::gen_scalar_type() {
  std::uniform_int_distribution<int> distr((int)ScalarType::EnumFirst,
                                           (int)ScalarType::EnumLast);
  return (ScalarType)distr(rng_);
}

}  // namespace fuzzer
