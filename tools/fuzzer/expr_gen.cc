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

enum class AllowedTypeKind : unsigned char {
  EnumFirst,

  // An integer type or the boolean type.
  Int = EnumFirst,

  // A floating-point type.
  Float,

  // Any pointer type that can be dereferenced (i.e. `void*`, `const void*`,
  // etc. and the `0` null pointer constant don't count).
  Pointer,

  // Any `void*` pointer or the `0` null pointer constant.
  VoidPointerOrNullConstant,

  EnumLast = VoidPointerOrNullConstant,
};
inline constexpr size_t NUM_ALLOWED_TYPE_KINDS =
    (size_t)AllowedTypeKind::EnumLast + 1;
using AllowedTypeKinds = EnumBitset<AllowedTypeKind>;

// All type kinds that can be allowed in a boolean context.
static constexpr AllowedTypeKinds ALL_IN_BOOL_CTX = AllowedTypeKinds::all_set();
// All type kinds that can be allowed in an integer only context (i.e. no
// floating point values).
static constexpr AllowedTypeKinds INT_KINDS = AllowedTypeKind::Int;
// All type kinds that can be allowed in an floating point context (i.e. both
// integers and floating point values).
static constexpr AllowedTypeKinds FLOAT_KINDS = {AllowedTypeKind::Int,
                                                 AllowedTypeKind::Float};

using TypeToGen = std::variant<AllowedTypeKinds, Type>;

class TypeConstraints {
 public:
  const TypeToGen& gen_type() const { return type_; }
  bool must_be_lvalue() const { return must_be_lvalue_; }

  const AllowedTypeKinds* as_type_kinds() const {
    return std::get_if<AllowedTypeKinds>(&type_);
  }
  bool is_type_kinds() const {
    return std::holds_alternative<AllowedTypeKinds>(type_);
  }

  bool allows_kind(AllowedTypeKind kind) const {
    const auto* ptr = as_type_kinds();
    if (ptr == nullptr) {
      return false;
    }

    return (*ptr)[kind];
  }

  bool is_type() const { return std::holds_alternative<Type>(type_); }
  const Type* as_type() const { return std::get_if<Type>(&type_); }
  bool is_pointer_type() const { return std::holds_alternative<Type>(type_); }
  const PointerType* as_pointer_type() const {
    const auto* ptr = std::get_if<Type>(&type_);
    // Will return null if `ptr` is null.
    return std::get_if<PointerType>(ptr);
  }
  // Returns if the constraints in question request a `void*` type, ignoring any
  // cv-qualifiers.
  bool is_void_pointer_type() const {
    const auto* ptr = as_pointer_type();
    if (ptr == nullptr) {
      return false;
    }

    // We don't care about cv qualifiers
    const auto* type = std::get_if<ScalarType>(&ptr->type().type());
    return type != nullptr && *type == ScalarType::Void;
  }

  const ScalarType* as_scalar_type() const {
    const auto* ptr = std::get_if<Type>(&type_);
    // Will return null if `ptr` is null.
    return std::get_if<ScalarType>(ptr);
  }
  bool is_any_scalar_type() const { return as_scalar_type() != nullptr; }
  bool is_nonvoid_scalar_type() const {
    const auto* ptr = as_scalar_type();
    return ptr != nullptr && *ptr != ScalarType::Void;
  }
  bool is_int_scalar_type() const {
    const auto* ptr = as_scalar_type();
    return ptr != nullptr && fuzzer::is_int_scalar_type(*ptr);
  }
  bool is_float_scalar_type() const {
    const auto* ptr = as_scalar_type();
    return ptr != nullptr && fuzzer::is_float_scalar_type(*ptr);
  }
  bool is_scalar_type(ScalarType scalar) const {
    const auto* ptr = as_scalar_type();
    return ptr != nullptr && *ptr == scalar;
  }

  const TaggedType* as_tagged_type() const {
    const auto* ptr = std::get_if<Type>(&type_);
    // Will return null if `ptr` is null.
    return std::get_if<TaggedType>(ptr);
  }
  bool is_tagged_type() const { return as_tagged_type() != nullptr; }

  TypeConstraints() = default;
  explicit TypeConstraints(TypeToGen type, bool must_be_lvalue = false)
      : type_(std::move(type)), must_be_lvalue_(must_be_lvalue) {}

 private:
  TypeToGen type_;
  bool must_be_lvalue_ = false;
};

static constexpr ExprKindMask LVALUE_KINDS = {{
    ExprKind::VariableExpr,
    ExprKind::DereferenceExpr,
    ExprKind::ArrayIndex,
    ExprKind::MemberOf,
    // `&(true ? x : y)` is semantically valid in C/C++
    ExprKind::MemberOfPtr,
}};

int expr_precedence(const Expr& e) {
  return std::visit([](const auto& e) { return e.precedence(); }, e);
}

std::optional<Expr> ExprGenerator::gen_boolean_constant(
    const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  if (constraints.allows_kind(AllowedTypeKind::Int) ||
      constraints.is_int_scalar_type()) {
    return BooleanConstant(rng_->gen_boolean());
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_integer_constant(
    const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  if (constraints.allows_kind(AllowedTypeKind::Int) ||
      constraints.is_nonvoid_scalar_type()) {
    return rng_->gen_integer_constant(cfg_.int_const_min, cfg_.int_const_max);
  }

  if (constraints.allows_kind(AllowedTypeKind::VoidPointerOrNullConstant) ||
      constraints.is_pointer_type()) {
    return IntegerConstant(0);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_double_constant(
    const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  if (constraints.allows_kind(AllowedTypeKind::Float) ||
      constraints.is_nonvoid_scalar_type()) {
    return rng_->gen_double_constant(cfg_.double_constant_min,
                                     cfg_.double_constant_max);
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_variable_expr(
    const TypeConstraints& constraints) {
  if (constraints.allows_kind(AllowedTypeKind::Int)) {
    return VariableExpr("x");
  }
  if (constraints.allows_kind(AllowedTypeKind::Float)) {
    return VariableExpr("fnan");
  }
  if (constraints.allows_kind(AllowedTypeKind::Pointer)) {
    return VariableExpr("null_char_ptr");
  }
  if (constraints.allows_kind(AllowedTypeKind::VoidPointerOrNullConstant)) {
    return VariableExpr("void_ptr");
  }

  auto* type_ptr = constraints.as_type();
  assert(type_ptr != nullptr &&
         "Not a type, did you introduce a new type kind that's not being "
         "handled?");

  const auto& symtab = cfg_.symbol_table;
  auto res = symtab.find(*type_ptr);
  assert(res != symtab.end() && "Could not find type in symbol table");
  assert(!res->second.empty() && "Could not find variables with this type");

  return VariableExpr(res->second[0]);
}

std::optional<Expr> ExprGenerator::gen_binary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  // We can't do pointer arithmetic with `void*`.
  if (constraints.is_void_pointer_type()) {
    return {};
  }

  BinOpMask mask = cfg_.bin_op_mask;

  // If we're expected to return a pointer type expression, we can only:
  // - Add or subtract pointers with integers.
  // - Subtract two non-void pointers of the same type (aka `ptrdiff_t`).
  // - Compare pointers. In C, all pointer types can be freely compared
  // against each other. C++, however, allows only the following:
  //   + Comparison of pointers to `void*` pointers or the `0` null pointer
  //   constant.
  //   + Comparison of pointers of the same type.
  //   + Comparison of pointers with a type in a superclass/subclass
  //   relationship (note the edge case regarding non-virtual multiple
  //   inheritance).
  static constexpr BinOpMask POINTER_OP_MASK = {
      BinOp::Plus,       BinOp::Minus,    BinOp::Eq, BinOp::Ne,
      BinOp::Lt,         BinOp::Le,       BinOp::Gt, BinOp::Ge,
      BinOp::LogicalAnd, BinOp::LogicalOr};
  if (constraints.is_pointer_type()) {
    mask &= POINTER_OP_MASK;
  }

  // If we're expected to return a floating point value, we can't use bitwise
  // operators and the modulo operator '%'.
  static constexpr BinOpMask FLOATING_POINT_OP_MASK = {
      BinOp::Plus, BinOp::Minus, BinOp::Mult,       BinOp::Div,
      BinOp::Eq,   BinOp::Ne,    BinOp::Lt,         BinOp::Le,
      BinOp::Gt,   BinOp::Ge,    BinOp::LogicalAnd, BinOp::LogicalOr};
  if (constraints.is_float_scalar_type()) {
    mask &= FLOATING_POINT_OP_MASK;
  }

  auto op = rng_->gen_bin_op(mask);

  auto maybe_lhs = gen_with_weights(weights, constraints);
  if (!maybe_lhs.has_value()) {
    return {};
  }
  auto&& lhs_rvalue = maybe_lhs.value();

  auto maybe_rhs = gen_with_weights(weights, constraints);
  if (!maybe_rhs.has_value()) {
    return {};
  }
  auto&& rhs_rvalue = maybe_rhs.value();

  // Rules for parenthesising the left hand side:
  // 1. If the left hand side has a strictly lower precedence than ours,
  //    then we will have to emit parens.
  //    Example: We emit `(3 + 4) * 5` instead of `3 + 4 * 5`.
  // 2. If the left hand side has the same precedence as we do, then we
  //    don't have to emit any parens. This is because all lldb-eval
  //    binary operators have left-to-right associativity.
  //    Example: We do not have to emit `(3 - 4) + 5`, `3 - 4 + 5` will also
  //    do.
  Expr lhs;
  if (expr_precedence(lhs_rvalue) > bin_op_precedence(op)) {
    lhs = ParenthesizedExpr(lhs_rvalue);
  } else {
    lhs = lhs_rvalue;
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
  Expr rhs;
  if (expr_precedence(rhs_rvalue) >= bin_op_precedence(op)) {
    rhs = ParenthesizedExpr(rhs_rvalue);
  } else {
    rhs = rhs_rvalue;
  }

  return BinaryExpr(std::move(lhs), op, std::move(rhs));
}

std::optional<Expr> ExprGenerator::gen_unary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  // Unary expressions cannot generate pointer values
  if (constraints.must_be_lvalue() || constraints.is_pointer_type()) {
    return {};
  }

  const auto type_kinds_ptr = constraints.as_type_kinds();
  if (type_kinds_ptr != nullptr && (*type_kinds_ptr & ~FLOAT_KINDS).none()) {
    return {};
  }

  auto mask = cfg_.un_op_mask;
  while (mask.any()) {
    auto op = (UnOp)rng_->gen_un_op(mask);

    AllowedTypeKinds type_kinds = FLOAT_KINDS;
    if (op == UnOp::BitNot) {
      type_kinds = INT_KINDS;
    } else if (op == UnOp::LogicalNot) {
      type_kinds = ALL_IN_BOOL_CTX;
    }

    auto maybe_expr = gen_with_weights(weights, TypeConstraints(type_kinds));
    if (!maybe_expr.has_value()) {
      continue;
    }

    auto&& expr_rvalue = maybe_expr.value();
    Expr expr;
    if (expr_precedence(expr_rvalue) > UnaryExpr::PRECEDENCE) {
      expr = ParenthesizedExpr(expr_rvalue);
    } else {
      expr = expr_rvalue;
    }

    return UnaryExpr(op, std::move(expr));
  }

  return {};
}

std::optional<Expr> ExprGenerator::gen_ternary_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_cond = gen_with_weights(weights, TypeConstraints(ALL_IN_BOOL_CTX));
  if (!maybe_cond.has_value()) {
    return {};
  }
  auto&& cond_rvalue = maybe_cond.value();

  TypeConstraints new_constraints{AllowedTypeKinds()};

  // We want to be sure that the left-hand side and right-hand side that
  // we generate have compatible types.
  if (constraints.must_be_lvalue() && !constraints.is_type()) {
    auto new_type = gen_type(weights, constraints);
    if (!new_type.has_value()) {
      return {};
    }
    new_constraints = TypeConstraints(std::move(new_type.value()),
                                      constraints.must_be_lvalue());
  } else {
    new_constraints = constraints;
  }

  auto maybe_lhs = gen_with_weights(weights, new_constraints);
  if (!maybe_lhs.has_value()) {
    return {};
  }
  auto&& lhs_rvalue = maybe_lhs.value();

  auto maybe_rhs = gen_with_weights(weights, new_constraints);
  if (!maybe_rhs.has_value()) {
    return {};
  }
  auto&& rhs_rvalue = maybe_rhs.value();

  Expr cond;
  if (expr_precedence(cond_rvalue) == TernaryExpr::PRECEDENCE) {
    cond = ParenthesizedExpr(cond_rvalue);
  } else {
    cond = cond_rvalue;
  }

  return TernaryExpr(std::move(cond), lhs_rvalue, rhs_rvalue);
}

std::optional<Expr> ExprGenerator::gen_cast_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue() || constraints.is_tagged_type()) {
    return {};
  }

  auto maybe_type = gen_type(weights, constraints);
  if (!maybe_type.has_value()) {
    return {};
  }
  auto&& type_rvalue = maybe_type.value();

  AllowedTypeKinds new_kinds;
  if (std::holds_alternative<PointerType>(type_rvalue)) {
    new_kinds = {AllowedTypeKind::Pointer,
                 AllowedTypeKind::VoidPointerOrNullConstant};
  } else {
    new_kinds = FLOAT_KINDS;
  }
  TypeConstraints new_constraints(new_kinds);

  auto maybe_expr = gen_with_weights(weights, new_constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto&& expr_rvalue = maybe_expr.value();

  Expr expr;
  if (expr_precedence(expr_rvalue) > CastExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(expr_rvalue);
  } else {
    expr = expr_rvalue;
  }

  return CastExpr(std::move(type_rvalue), std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_address_of_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  if (constraints.must_be_lvalue()) {
    return {};
  }

  if (!constraints.allows_kind(AllowedTypeKind::Pointer) ||
      !constraints.is_pointer_type()) {
    return {};
  }

  TypeConstraints new_constraints;
  const auto* type_ptr = constraints.as_pointer_type();
  if (type_ptr == nullptr) {
    new_constraints = TypeConstraints(AllowedTypeKind::Pointer);
  } else {
    new_constraints = TypeConstraints(type_ptr->type().type());
  }

  auto maybe_expr = gen_with_weights(weights, new_constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto&& expr_rvalue = maybe_expr.value();

  Expr expr;
  if (expr_precedence(expr_rvalue) > AddressOf::PRECEDENCE) {
    expr = ParenthesizedExpr(expr_rvalue);
  } else {
    expr = expr_rvalue;
  }

  return AddressOf(std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_member_of_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto&& expr_rvalue = maybe_expr.value();

  Expr expr;
  if (expr_precedence(expr_rvalue) > MemberOf::PRECEDENCE) {
    expr = ParenthesizedExpr(expr_rvalue);
  } else {
    expr = expr_rvalue;
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
  TypeConstraints idx_constraints(INT_KINDS);

  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto&& expr_rvalue = maybe_expr.value();

  auto maybe_idx = gen_with_weights(weights, idx_constraints);
  if (!maybe_idx.has_value()) {
    return {};
  }
  auto&& idx_rvalue = maybe_idx.value();

  Expr expr;
  if (expr_precedence(expr_rvalue) > ArrayIndex::PRECEDENCE) {
    expr = ParenthesizedExpr(expr_rvalue);
  } else {
    expr = expr_rvalue;
  }

  return ArrayIndex(std::move(expr), idx_rvalue);
}

std::optional<Expr> ExprGenerator::gen_dereference_expr(
    const Weights& weights, const TypeConstraints& constraints) {
  auto maybe_expr = gen_with_weights(weights, constraints);
  if (!maybe_expr.has_value()) {
    return {};
  }
  auto&& expr_rvalue = maybe_expr.value();

  Expr expr;
  if (expr_precedence(expr_rvalue) > DereferenceExpr::PRECEDENCE) {
    expr = ParenthesizedExpr(expr_rvalue);
  } else {
    expr = expr_rvalue;
  }

  return DereferenceExpr(std::move(expr));
}

std::optional<Expr> ExprGenerator::gen_with_weights(
    const Weights& weights, const TypeConstraints& constraints) {
  Weights new_weights = weights;

  ExprKindMask mask = ExprKindMask::all_set();
  if (constraints.must_be_lvalue()) {
    mask &= LVALUE_KINDS;
  }

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
      mask[idx] = false;

      continue;
    }
    auto&& expr_rvalue = maybe_expr.value();

    return maybe_parenthesized(expr_rvalue);
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
  static constexpr TypeKindMask NO_TAGGED_TYPES = {TypeKind::ScalarType,
                                                   TypeKind::PointerType};

  return gen_type_impl(weights, constraints, NO_TAGGED_TYPES);
}

std::optional<Type> ExprGenerator::gen_type_impl(
    const Weights& weights, const TypeConstraints& constraints,
    TypeKindMask mask) {
  Weights new_weights = weights;

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
  auto maybe_type =
      gen_type_impl(weights, constraints, TypeKindMask::all_set());
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

  return gen_with_weights(weights, TypeConstraints(ALL_IN_BOOL_CTX));
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

ScalarType DefaultGeneratorRng::gen_scalar_type() {
  std::uniform_int_distribution<int> distr((int)ScalarType::EnumFirst,
                                           (int)ScalarType::EnumLast);
  return (ScalarType)distr(rng_);
}

}  // namespace fuzzer
