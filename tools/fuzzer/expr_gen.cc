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
#include <unordered_set>
#include <variant>

#include "lldb-eval/defines.h"
#include "tools/fuzzer/ast.h"
#include "tools/fuzzer/enum_bitset.h"

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

using ScalarMask = EnumBitset<ScalarType>;

static constexpr ScalarMask INT_TYPES = {
    ScalarType::Bool,           ScalarType::Char,
    ScalarType::UnsignedChar,   ScalarType::SignedChar,
    ScalarType::SignedShort,    ScalarType::UnsignedShort,
    ScalarType::SignedInt,      ScalarType::UnsignedInt,
    ScalarType::SignedLong,     ScalarType::UnsignedLong,
    ScalarType::SignedLongLong, ScalarType::UnsignedLongLong,
};

static constexpr ScalarMask FLOAT_TYPES = {
    ScalarType::Float,
    ScalarType::Double,
    ScalarType::LongDouble,
};

class NoType {};
class AnyType {};
class SpecificTypes {
 public:
  using PtrConstraintsType =
      std::variant<NoType, AnyType, std::shared_ptr<SpecificTypes>>;

  SpecificTypes() = default;
  SpecificTypes(ScalarMask scalar_types,
                std::unordered_set<TaggedType> tagged_types)
      : scalar_types_(std::move(scalar_types)),
        tagged_types_(std::move(tagged_types)) {}

  SpecificTypes(ScalarMask scalar_types)
      : scalar_types_(std::move(scalar_types)) {}

  SpecificTypes(const Type& type) {
    const auto* scalar_type = std::get_if<ScalarType>(&type);
    if (scalar_type != nullptr) {
      scalar_types_[*scalar_type] = true;
      return;
    }

    const auto* tagged_type = std::get_if<TaggedType>(&type);
    if (tagged_type != nullptr) {
      tagged_types_.insert(*tagged_type);
      return;
    }

    const auto* pointer_type = std::get_if<PointerType>(&type);
    if (pointer_type != nullptr) {
      ptr_types_ = std::make_shared<SpecificTypes>(pointer_type->type().type());
      return;
    }
  }

  explicit SpecificTypes(std::unordered_set<TaggedType> tagged_types)
      : tagged_types_(std::move(tagged_types)) {}

  SpecificTypes(NoType, bool allows_void_ptr_type = false)
      : ptr_types_(NoType()), allows_void_ptr_type_(allows_void_ptr_type) {}

  SpecificTypes(AnyType, bool allows_void_ptr_type = false)
      : ptr_types_(AnyType()), allows_void_ptr_type_(allows_void_ptr_type) {}

  SpecificTypes(SpecificTypes ptr_types, bool allows_void_ptr_type)
      : ptr_types_(std::make_shared<SpecificTypes>(std::move(ptr_types))),
        allows_void_ptr_type_(allows_void_ptr_type) {}

  SpecificTypes(ScalarMask scalar_types, SpecificTypes ptr_types,
                bool allows_void_ptr_type = false)
      : scalar_types_(std::move(scalar_types)),
        ptr_types_(std::make_shared<SpecificTypes>(std::move(ptr_types))),
        allows_void_ptr_type_(allows_void_ptr_type) {}

  SpecificTypes(ScalarMask scalar_types,
                std::unordered_set<TaggedType> tagged_types,
                SpecificTypes ptr_types, bool allows_void_ptr_type = false)
      : scalar_types_(std::move(scalar_types)),
        tagged_types_(std::move(tagged_types)),
        ptr_types_(std::make_shared<SpecificTypes>(std::move(ptr_types))),
        allows_void_ptr_type_(allows_void_ptr_type) {}

  SpecificTypes operator&(const SpecificTypes& rhs) const {
    SpecificTypes retval;
    retval.scalar_types_ = scalar_types_ & rhs.scalar_types_;
    for (const auto& e : rhs.tagged_types_) {
      if (tagged_types_.find(e) != tagged_types_.end()) {
        retval.tagged_types_.insert(e);
      }
    }
    retval.allows_void_ptr_type_ =
        allows_void_ptr_type_ & rhs.allows_void_ptr_type_;

    if (std::holds_alternative<NoType>(ptr_types_) ||
        std::holds_alternative<NoType>(rhs.ptr_types_)) {
      retval.ptr_types_ = NoType();
    } else if (std::holds_alternative<AnyType>(ptr_types_)) {
      retval.ptr_types_ = rhs.ptr_types_;
    } else if (std::holds_alternative<AnyType>(rhs.ptr_types_)) {
      retval.ptr_types_ = ptr_types_;
    } else {
      const auto* lhs_ptr =
          std::get_if<std::shared_ptr<SpecificTypes>>(&ptr_types_);
      const auto* rhs_ptr =
          std::get_if<std::shared_ptr<SpecificTypes>>(&rhs.ptr_types_);

      assert(lhs_ptr != nullptr && rhs_ptr != nullptr);
      assert(*lhs_ptr != nullptr && *rhs_ptr != nullptr);

      retval.ptr_types_ =
          std::make_shared<SpecificTypes>(**lhs_ptr & **rhs_ptr);
    }

    return retval;
  }

  ScalarMask scalar_types() const { return scalar_types_; }

  const std::unordered_set<TaggedType>& tagged_types() const {
    return tagged_types_;
  }

  bool allows_void_ptr() const { return allows_void_ptr_type_; }

  bool any_ptr_type() const {
    return std::holds_alternative<AnyType>(ptr_types_) && allows_void_ptr_type_;
  }

  bool no_ptr_type() const {
    return std::holds_alternative<NoType>(ptr_types_) && !allows_void_ptr_type_;
  }

  bool allows_type(const Type& type) const {
    const auto* as_scalar = std::get_if<ScalarType>(&type);
    if (as_scalar != nullptr) {
      return scalar_types_[*as_scalar];
    }

    const auto* as_tagged = std::get_if<TaggedType>(&type);
    if (as_tagged != nullptr) {
      return tagged_types_.find(*as_tagged) != tagged_types_.end();
    }

    const auto* as_ptr = std::get_if<PointerType>(&type);
    if (allows_void_ptr_type_) {
      const auto* as_ptr_to_scalar =
          std::get_if<ScalarType>(&as_ptr->type().type());
      if (as_ptr_to_scalar != nullptr &&
          *as_ptr_to_scalar == ScalarType::Void) {
        return true;
      }
    }

    if (std::holds_alternative<AnyType>(ptr_types_)) {
      return true;
    }

    if (std::holds_alternative<NoType>(ptr_types_)) {
      return false;
    }

    const auto* inner =
        std::get_if<std::shared_ptr<SpecificTypes>>(&ptr_types_);

    assert(inner != nullptr);
    assert(*inner != nullptr);
    assert(as_ptr != nullptr);

    return (*inner)->allows_type(*as_ptr);
  }

  bool satisfiable() const {
    return scalar_types_.any() || !tagged_types_.empty() ||
           !std::holds_alternative<NoType>(ptr_types_) || allows_void_ptr_type_;
  }

  const PtrConstraintsType& ptr_types() const { return ptr_types_; }

 private:
  ScalarMask scalar_types_;
  std::unordered_set<TaggedType> tagged_types_;
  PtrConstraintsType ptr_types_;
  bool allows_void_ptr_type_ = false;
};

SpecificTypes all_in_bool_ctx() {
  return SpecificTypes(INT_TYPES | FLOAT_TYPES, AnyType(), true);
}

class ExprConstraints {
 public:
  ExprConstraints() = default;
  ExprConstraints(NoType, bool must_be_lvalue = false)
      : constraints_(NoType()), must_be_lvalue_(must_be_lvalue) {}
  ExprConstraints(AnyType, bool must_be_lvalue = false)
      : constraints_(AnyType()), must_be_lvalue_(must_be_lvalue) {}
  ExprConstraints(SpecificTypes types, bool must_be_lvalue = false)
      : constraints_(std::move(types)), must_be_lvalue_(must_be_lvalue) {}

  const ConstraintsType& constraints() const { return constraints_; }

  bool must_be_lvalue() const { return must_be_lvalue_; }

  bool satisfiable() const {
    if (std::holds_alternative<NoType>(constraints_)) {
      return false;
    }

    if (std::holds_alternative<AnyType>(constraints_)) {
      return true;
    }

    const auto* as_specific_types = std::get_if<SpecificTypes>(&constraints_);
    return as_specific_types->satisfiable();
  }

  bool allows_type(const Type& type) const {
    if (std::holds_alternative<NoType>(constraints_)) {
      return false;
    }

    if (std::holds_alternative<AnyType>(constraints_)) {
      return true;
    }

    return std::get_if<SpecificTypes>(&constraints_)->allows_type(type);
  }

  ExprConstraints operator&(const ExprConstraints& rhs) const {
    ExprConstraints retval;
    retval.must_be_lvalue_ = must_be_lvalue_ | rhs.must_be_lvalue_;

    if (std::holds_alternative<NoType>(constraints_) ||
        std::holds_alternative<NoType>(rhs.constraints_)) {
      retval.constraints_ = NoType();
    } else if (std::holds_alternative<AnyType>(constraints_)) {
      retval.constraints_ = rhs.constraints_;
    } else if (std::holds_alternative<AnyType>(rhs.constraints_)) {
      retval.constraints_ = constraints_;
    } else {
      const auto* lhs_ptr = std::get_if<SpecificTypes>(&constraints_);
      const auto* rhs_ptr = std::get_if<SpecificTypes>(&rhs.constraints_);

      retval.constraints_ = *lhs_ptr & *rhs_ptr;
    }

    return retval;
  }

 private:
  std::variant<NoType, AnyType, SpecificTypes> constraints_;
  bool must_be_lvalue_ = false;
};

int expr_precedence(const Expr& e) {
  return std::visit([](const auto& e) { return e.precedence(); }, e);
}

std::optional<Expr> ExprGenerator::gen_boolean_constant(
    const ExprConstraints& constraints) {
  auto bool_constraints = constraints & ExprConstraints(INT_TYPES);
  if (bool_constraints.must_be_lvalue() || !bool_constraints.satisfiable()) {
    return {};
  }

  return BooleanConstant(rng_->gen_boolean());
}

std::optional<Expr> ExprGenerator::gen_integer_constant(
    const ExprConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  // Integers can be generated in place of floats
  auto int_constraints = constraints & ExprConstraints(INT_TYPES | FLOAT_TYPES);
  if (int_constraints.satisfiable()) {
    return rng_->gen_integer_constant(cfg_.int_const_min, cfg_.int_const_max);
  }

  auto voidptr_constraints = constraints & ExprConstraints(NoType(), true);
  if (voidptr_constraints.satisfiable()) {
    return IntegerConstant(0);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_double_constant(
    const ExprConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  auto float_constraints = constraints & ExprConstraints(FLOAT_TYPES);
  if (float_constraints.satisfiable()) {
    return rng_->gen_double_constant(cfg_.double_constant_min,
                                     cfg_.double_constant_max);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_variable_expr(
    const ExprConstraints& constraints) {
  for (const auto& type : cfg_.symbol_table) {
    if (!constraints.allows_type(type.first)) {
      continue;
    }

    if (type.second.empty()) {
      continue;
    }

    return VariableExpr(type.second[0]);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_binary_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  BinOpMask mask = cfg_.bin_op_mask;

  // We can't use floating point operands with these operators.
  static constexpr BinOpMask INT_ONLY_OPS = {BinOp::BitAnd, BinOp::BitOr,
                                             BinOp::BitXor, BinOp::Shl,
                                             BinOp::Shr,    BinOp::Mod};

  static constexpr BinOpMask CMP_OPS = {
      BinOp::Eq, BinOp::Ne, BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge,
  };
  static constexpr BinOpMask LOGICAL_OPS = {BinOp::LogicalAnd,
                                            BinOp::LogicalOr};
  ScalarMask allow_floats_mask;
  if ((constraints & SpecificTypes(FLOAT_TYPES)).satisfiable()) {
    allow_floats_mask = FLOAT_TYPES;
  }

  while (mask.any()) {
    auto op = rng_->gen_bin_op(mask);

    SpecificTypes lhs_types;
    SpecificTypes rhs_types;
    SpecificTypes output_types;

    if (INT_ONLY_OPS[op]) {
      lhs_types = INT_TYPES;
      rhs_types = INT_TYPES;
      output_types = INT_TYPES;
    } else if (LOGICAL_OPS[op]) {
      lhs_types = all_in_bool_ctx();
      rhs_types = all_in_bool_ctx();
      output_types = ScalarMask(ScalarType::Bool);
    } else if (CMP_OPS[op]) {
      output_types = ScalarMask(ScalarType::Bool);

      if (rng_->gen_binop_ptr_expr(cfg_.binop_gen_ptr_expr_prob)) {
        SpecificTypes ptr_type(AnyType(), true);
        auto maybe_type = gen_type(
            weights, constraints & ExprConstraints(std::move(ptr_type)));
        if (!maybe_type.has_value()) {
          mask[op] = false;
          continue;
        }
        Type type = std::move(maybe_type.value());

        lhs_types = type;
        rhs_types = type;
      } else {
        lhs_types = INT_TYPES | allow_floats_mask;
        rhs_types = INT_TYPES | allow_floats_mask;
      }
    } else if (op == BinOp::Plus) {
      if (rng_->gen_binop_ptr_expr(cfg_.binop_gen_ptr_expr_prob)) {
        lhs_types = SpecificTypes(AnyType());
        rhs_types = INT_TYPES;
        output_types = SpecificTypes(AnyType());
      } else {
        lhs_types = INT_TYPES | allow_floats_mask;
        rhs_types = INT_TYPES | allow_floats_mask;
        output_types = INT_TYPES | allow_floats_mask;
      }
    } else if (op == BinOp::Minus) {
      if (rng_->gen_binop_ptr_expr(cfg_.binop_gen_ptr_expr_prob)) {
        lhs_types = SpecificTypes(AnyType());
        rhs_types = INT_TYPES;
        output_types = SpecificTypes(AnyType());
      } else if (rng_->gen_binop_ptrdiff_expr(
                     cfg_.binop_gen_ptrdiff_expr_prob)) {
        SpecificTypes ptr_type(AnyType(), true);
        auto maybe_type = gen_type(
            weights, constraints & ExprConstraints(std::move(ptr_type)));
        if (!maybe_type.has_value()) {
          mask[op] = false;
          continue;
        }
        Type type = std::move(maybe_type.value());

        lhs_types = type;
        rhs_types = type;
      } else {
        lhs_types = INT_TYPES | allow_floats_mask;
        rhs_types = INT_TYPES | allow_floats_mask;
        output_types = INT_TYPES | allow_floats_mask;
      }
    } else {
      lhs_types = INT_TYPES | allow_floats_mask;
      rhs_types = INT_TYPES | allow_floats_mask;
      output_types = INT_TYPES | allow_floats_mask;
    }

    if (rng_->gen_binop_flip_operands(cfg_.binop_flip_operands_prob)) {
      std::swap(lhs_types, rhs_types);
    }

    auto lhs_constraints = constraints & lhs_types;
    auto rhs_constraints = constraints & rhs_types;
    auto output_constraints = constraints & output_types;
    if (!lhs_constraints.satisfiable() || !rhs_constraints.satisfiable() ||
        !output_constraints.satisfiable()) {
      mask[op] = false;
      continue;
    }

    auto maybe_lhs = gen_with_weights(weights, std::move(lhs_constraints));
    if (!maybe_lhs.has_value()) {
      mask[op] = false;
      continue;
    }
    Expr lhs = std::move(maybe_lhs.value());

    auto maybe_rhs = gen_with_weights(weights, std::move(rhs_constraints));
    if (!maybe_rhs.has_value()) {
      mask[op] = false;
      continue;
    }
    Expr rhs = std::move(maybe_rhs.value());

    // Rules for parenthesising the left hand side:
    // 1. If the left hand side has a strictly lower precedence than ours,
    //    then we will have to emit parens.
    //    Example: We emit `(3 + 4) * 5` instead of `3 + 4 * 5`.
    // 2. If the left hand side has the same precedence as we do, then we
    //    don't have to emit any parens. This is because all lldb-eval
    //    binary operators have left-to-right associativity.
    //    Example: We do not have to emit `(3 - 4) + 5`, `3 - 4 + 5` will also
    //    do.
    if (expr_precedence(lhs) > bin_op_precedence(op)) {
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
    if (expr_precedence(rhs) >= bin_op_precedence(op)) {
      rhs = ParenthesizedExpr(std::move(rhs));
    }

    return BinaryExpr(std::move(lhs), op, std::move(rhs));
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_unary_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  UnOpMask mask = cfg_.un_op_mask;

  ScalarMask allow_floats_mask = FLOAT_TYPES;
  auto float_constraints = constraints & ExprConstraints(FLOAT_TYPES);
  if (!float_constraints.satisfiable()) {
    mask[UnOp::BitNot] = false;
    allow_floats_mask = ScalarMask();
  }

  auto int_constraints = constraints & ExprConstraints(INT_TYPES);
  if (!int_constraints.satisfiable()) {
    mask = UnOp::LogicalNot;
  }

  while (mask.any()) {
    auto op = (UnOp)rng_->gen_un_op(mask);

    SpecificTypes input_types;

    SpecificTypes output_types;
    if (op == UnOp::LogicalNot) {
      input_types = all_in_bool_ctx();
      output_types = ScalarMask(ScalarType::Bool);
    } else if (op == UnOp::BitNot) {
      input_types = INT_TYPES;
      output_types = INT_TYPES;
    } else {
      input_types = INT_TYPES | allow_floats_mask;
      output_types = INT_TYPES | allow_floats_mask;
    }

    auto input_intersection = constraints & input_types;
    auto output_constraints = constraints & output_types;
    if (!output_constraints.satisfiable()) {
      return {};
    }

    auto maybe_expr = gen_with_weights(weights, std::move(input_types));
    if (!maybe_expr.has_value()) {
      mask[op] = false;
      continue;
    }
    Expr expr = std::move(maybe_expr.value());

    if (expr_precedence(expr) > UnaryExpr::PRECEDENCE) {
      expr = ParenthesizedExpr(expr);
    }

    return UnaryExpr(op, std::move(expr));
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_ternary_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  auto maybe_cond =
      gen_with_weights(weights, ExprConstraints(all_in_bool_ctx()));
  if (!maybe_cond.has_value()) {
    return {};
  }
  Expr cond = std::move(maybe_cond.value());

  ExprConstraints new_constraints;
  SpecificTypes value_types = INT_TYPES | FLOAT_TYPES;
  if (!(constraints & SpecificTypes(FLOAT_TYPES)).satisfiable()) {
    value_types = INT_TYPES;
  }

  if (constraints.must_be_lvalue() ||
      !(constraints & ExprConstraints(value_types)).satisfiable()) {
    auto maybe_type = gen_type(weights, constraints);
    if (!maybe_type.has_value()) {
      return {};
    }
    Type type = std::move(maybe_type.value());
    new_constraints = ExprConstraints(type, constraints.must_be_lvalue());
  } else {
    new_constraints = value_types;
  }

  auto maybe_lhs = gen_with_weights(weights, new_constraints);
  if (!maybe_lhs.has_value()) {
    return {};
  }
  Expr lhs = std::move(maybe_lhs.value());

  auto maybe_rhs = gen_with_weights(weights, new_constraints);
  if (!maybe_rhs.has_value()) {
    return {};
  }
  Expr rhs = std::move(maybe_rhs.value());

  if (expr_precedence(cond) == TernaryExpr::PRECEDENCE) {
    cond = ParenthesizedExpr(cond);
  }

  return TernaryExpr(std::move(cond), std::move(lhs), std::move(rhs));
}

std::optional<Expr> ExprGenerator::gen_cast_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  auto maybe_type =
      gen_type(weights, constraints & SpecificTypes(INT_TYPES | FLOAT_TYPES, {},
                                                    AnyType{}, true));
  if (!maybe_type.has_value()) {
    return {};
  }
  Type type = std::move(maybe_type.value());
  SpecificTypes new_constraints;
  if (std::holds_alternative<PointerType>(type)) {
    new_constraints = AnyType();
  } else {
    new_constraints = INT_TYPES | FLOAT_TYPES;
  }

  auto maybe_expr = gen_with_weights(weights, std::move(new_constraints));
  if (!maybe_expr.has_value()) {
    return {};
  }
  Expr expr = std::move(maybe_expr.value());

  if (expr_precedence(expr) > CastExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return CastExpr(std::move(type), std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_address_of_expr(
    const Weights&, const ExprConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  return {};

  // ExprConstraints new_constraints;
  // if (type_ptr == nullptr) {
  //   new_constraints = ExprConstraints(AllowedTypeKind::Pointer);
  // } else {
  //   new_constraints = ExprConstraints(type_ptr->type().type());
  // }

  // auto maybe_expr = gen_with_weights(weights, new_constraints);
  // if (!maybe_expr.has_value()) {
  //   return {};
  // }
  // Expr expr = std::move(maybe_expr.value());

  // if (expr_precedence(expr) > AddressOf::PRECEDENCE) {
  //   expr = ParenthesizedExpr(std::move(expr));
  // }

  // return AddressOf(std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_member_of_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  Expr expr = maybe_expr.value();

  if (expr_precedence(expr) > MemberOf::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return MemberOf(std::move(expr), "f1");
}

std::optional<Expr> ExprGenerator::gen_member_of_ptr_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  Expr expr = std::move(maybe_expr.value());

  if (expr_precedence(expr) > MemberOfPtr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return MemberOfPtr(std::move(expr), "f1");
}

std::optional<Expr> ExprGenerator::gen_array_index_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  ExprConstraints idx_constraints = SpecificTypes(INT_TYPES);

  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  Expr expr = std::move(maybe_expr.value());

  auto maybe_idx = gen_with_weights(weights, idx_constraints);
  if (!maybe_idx.has_value()) {
    return {};
  }
  Expr idx = std::move(maybe_idx.value());

  if (expr_precedence(expr) > ArrayIndex::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return ArrayIndex(std::move(expr), std::move(idx));
}

std::optional<Expr> ExprGenerator::gen_dereference_expr(
    const Weights& weights, const ExprConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  Expr expr = std::move(maybe_expr.value());

  if (expr_precedence(expr) > DereferenceExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(std::move(expr));
  }

  return DereferenceExpr(std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_with_weights(
    const Weights& weights, const ExprConstraints& constraints) {
  if (!constraints.satisfiable()) {
    return {};
  }

  Weights new_weights = weights;

  ExprKindMask mask = ExprKindMask::all_set();
  while (mask.any()) {
    auto kind = rng_->gen_expr_kind(new_weights, mask);
    auto idx = (size_t)kind;

    if (!mask[kind]) {
      break;
    }

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

      case ExprKind::DereferenceExpr:
        maybe_expr = gen_dereference_expr(new_weights, constraints);
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
      mask[kind] = false;

      continue;
    }

    return maybe_parenthesized(std::move(maybe_expr.value()));
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
    const Weights& weights, const ExprConstraints& constraints) {
  return gen_type_impl(weights, constraints.constraints());
}

std::optional<Type> ExprGenerator::gen_type_impl(
    const Weights& weights, const ConstraintsType& type_constraints) {
  if (std::holds_alternative<NoType>(type_constraints)) {
    return {};
  }
  const auto* as_specific_types = std::get_if<SpecificTypes>(&type_constraints);

  Weights new_weights = weights;
  TypeKindMask mask = TypeKindMask::all_set();
  if (as_specific_types != nullptr) {
    if (as_specific_types->scalar_types().none()) {
      mask[TypeKind::ScalarType] = false;
    }
    if (as_specific_types->tagged_types().empty()) {
      mask[TypeKind::TaggedType] = false;
    }
    if (!as_specific_types->any_ptr_type() ||
        !as_specific_types->allows_void_ptr()) {
      mask[TypeKind::TaggedType] = false;
    }
  }

  while (mask.any()) {
    auto choice = rng_->gen_type_kind(new_weights, mask);
    auto idx = (size_t)choice;

    auto& new_type_weights = new_weights.type_weights();
    auto old_weight = new_type_weights[idx];
    new_type_weights[idx] *= cfg_.type_kind_weights[idx].dampening_factor;

    std::optional<Type> maybe_type;
    switch (choice) {
      case TypeKind::ScalarType:
        maybe_type = gen_scalar_type(type_constraints);
        break;

      case TypeKind::TaggedType:
        maybe_type = gen_tagged_type(type_constraints);
        break;

      case TypeKind::PointerType:
        maybe_type = gen_pointer_type(new_weights, type_constraints);
        break;
    }

    if (maybe_type.has_value()) {
      return maybe_type;
    }

    new_type_weights[idx] = old_weight;
    mask[choice] = false;
  }

  return {};
}

std::optional<QualifiedType> ExprGenerator::gen_qualified_type(
    const Weights& weights, const ConstraintsType& constraints) {
  auto maybe_type = gen_type_impl(weights, constraints);
  if (!maybe_type.has_value()) {
    return {};
  }
  Type type = std::move(maybe_type.value());
  auto qualifiers = gen_cv_qualifiers();

  return QualifiedType(std::move(type), qualifiers);
}

std::optional<Type> ExprGenerator::gen_pointer_type(
    const Weights& weights, const ConstraintsType& constraints) {
  ConstraintsType new_constraints;
  const auto* specific_types = std::get_if<SpecificTypes>(&constraints);
  if (specific_types != nullptr) {
    const auto* ptr_constraints = std::get_if<std::shared_ptr<SpecificTypes>>(
        &specific_types->ptr_types());
    if (ptr_constraints != nullptr) {
      new_constraints = **ptr_constraints;
    }
  } else {
    new_constraints = constraints;
  }

  auto maybe_type = gen_qualified_type(weights, new_constraints);
  if (!maybe_type.has_value()) {
    return {};
  }
  auto& type = maybe_type.value();

  return PointerType(std::move(type));
}

std::optional<Type> ExprGenerator::gen_tagged_type(
    const ConstraintsType& constraints) {
  if (std::holds_alternative<NoType>(constraints)) {
    return {};
  }

  const SpecificTypes* as_tagged_types =
      std::get_if<SpecificTypes>(&constraints);
  if (as_tagged_types != nullptr) {
    auto it = as_tagged_types->tagged_types().begin();
    if (it != as_tagged_types->tagged_types().end()) {
      return *it;
    }
  }

  return {};
}

std::optional<Type> ExprGenerator::gen_scalar_type(
    const ConstraintsType& constraints) {
  ScalarMask mask;
  if (std::holds_alternative<AnyType>(constraints)) {
    mask = ScalarMask::all_set();
  } else {
    const SpecificTypes* as_types = std::get_if<SpecificTypes>(&constraints);
    if (as_types == nullptr) {
      return {};
    }

    mask = as_types->scalar_types();
  }

  return rng_->gen_scalar_type(mask);
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

  return gen_with_weights(weights, ExprConstraints(all_in_bool_ctx()));
}

template <typename Enum, typename Rng>
Enum pick_nth_set_bit(const EnumBitset<Enum> mask, Rng& rng) {
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
      return (Enum)i;
    }
  }

  // `choice` lies in the range `[1, mask.count()]`, `running_ones` will
  // always lie in the range `[0, mask.count()]` and is incremented at most
  // once per loop iteration. The only way for this assertion to fire is for
  // `mask` to be empty (which we have asserted beforehand).
  lldb_eval_unreachable("Mask has no bits set");
}

template <typename Enum, typename Rng, typename RealType>
Enum weighted_pick(
    const std::array<RealType, (size_t)Enum::EnumLast + 1>& array,
    const EnumBitset<Enum>& mask, Rng& rng) {
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
      return (Enum)i;
    }
  }

  // Just in case we get here due to e.g. floating point inaccuracies, etc.
  return Enum::EnumLast;
}

BinOp DefaultGeneratorRng::gen_bin_op(BinOpMask mask) {
  return pick_nth_set_bit(mask, rng_);
}

UnOp DefaultGeneratorRng::gen_un_op(UnOpMask mask) {
  return pick_nth_set_bit(mask, rng_);
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

  CvQualifiers retval;
  retval[CvQualifier::Const] = const_distr(rng_);
  retval[CvQualifier::Volatile] = volatile_distr(rng_);

  return retval;
}

bool DefaultGeneratorRng::gen_binop_ptr_expr(float probability) {
  std::bernoulli_distribution distr(probability);
  return distr(rng_);
}

bool DefaultGeneratorRng::gen_binop_flip_operands(float probability) {
  std::bernoulli_distribution distr(probability);
  return distr(rng_);
}

bool DefaultGeneratorRng::gen_binop_ptrdiff_expr(float probability) {
  std::bernoulli_distribution distr(probability);
  return distr(rng_);
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
  return weighted_pick(weights.expr_weights(), mask, rng_);
}

TypeKind DefaultGeneratorRng::gen_type_kind(const Weights& weights,
                                            const TypeKindMask& mask) {
  return weighted_pick(weights.type_weights(), mask, rng_);
}

ScalarType DefaultGeneratorRng::gen_scalar_type(ScalarMask mask) {
  return pick_nth_set_bit(mask, rng_);
}

}  // namespace fuzzer
