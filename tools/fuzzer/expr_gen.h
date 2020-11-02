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

#ifndef INCLUDE_EXPR_GEN_H
#define INCLUDE_EXPR_GEN_H

#include <array>
#include <bitset>
#include <cstdint>
#include <random>

#include "tools/fuzzer/ast.h"

namespace fuzzer {

enum class ExprKind : unsigned char {
  EnumFirst,
  IntegerConstant = EnumFirst,
  DoubleConstant,
  VariableExpr,
  UnaryExpr,
  BinaryExpr,
  AddressOf,
  MemberOf,
  MemberOfPtr,
  ArrayIndex,
  TernaryExpr,
  BooleanConstant,
  CastExpr,
  EnumLast = CastExpr,
};
inline constexpr size_t NUM_GEN_EXPR_KINDS = (size_t)ExprKind::EnumLast + 1;

enum class ScalarConstraints : unsigned char { AllowInts, AllowFloats };

class Weights;
class TypeConstraints;

struct ExprKindWeightInfo {
  float initial_weight;
  float dampening_factor;
};

struct TypeKindWeightInfo {
  float initial_weight;
  float dampening_factor;
};

using BinOpMask = std::bitset<NUM_BIN_OPS>;
using UnOpMask = std::bitset<NUM_UN_OPS>;

struct GenConfig {
  int num_exprs_to_generate = 20;

  uint64_t int_const_min = 0;
  uint64_t int_const_max = 1000;

  double double_constant_min = 0;
  double double_constant_max = 10;

  float parenthesize_prob = 0.2f;

  float const_prob = 0.3f;
  float volatile_prob = 0.05f;

  BinOpMask bin_op_mask = ~0ull;
  UnOpMask un_op_mask = ~0ull;

  std::array<ExprKindWeightInfo, NUM_EXPR_KINDS> expr_kind_weights = {{
      {1.0f, 0.0f},  // ExprKind::IntegerConstant
      {2.0f, 0.0f},  // ExprKind::DoubleConstant
      {1.0f, 0.0f},  // ExprKind::VariableExpr
      {7.0f, 0.4f},  // ExprKind::UnaryExpr
      {3.0f, 0.4f},  // ExprKind::BinaryExpr
      {1.0f, 0.1f},  // ExprKind::AddressOf
      {1.0f, 0.1f},  // ExprKind::MemberOf
      {1.0f, 0.1f},  // ExprKind::MemberOfPtr
      {1.0f, 0.1f},  // ExprKind::ArrayIndex
      {2.0f, 0.1f},  // ExprKind::TernaryExpr
      {1.0f, 0.0f},  // ExprKind::BooleanConstant
      {1.0f, 0.4f},  // ExprKind::CastExpr
  }};

  std::array<TypeKindWeightInfo, NUM_GEN_TYPE_KINDS> type_kind_weights = {{
      {2.0f, 0.0f},  // TypeKind::ScalarType
      {1.0f, 0.0f},  // TypeKind::TaggedType
      {1.0f, 0.1f},  // TypeKind::PointerType
      {1.0f, 0.1f},  // TypeKind::ReferenceType
  }};
};

class GeneratorRng {
 public:
  virtual ~GeneratorRng() {}

  virtual BinOp gen_bin_op(BinOpMask mask) = 0;
  virtual UnOp gen_un_op(UnOpMask mask) = 0;
  virtual ExprKind gen_expr_kind(const Weights& array) = 0;
  virtual TypeKind gen_type_kind(const Weights& array) = 0;
  virtual ScalarType gen_scalar_type() = 0;
  virtual bool gen_boolean() = 0;
  virtual uint64_t gen_u64(uint64_t min, uint64_t max) = 0;
  virtual double gen_double(double min, double max) = 0;
  virtual bool gen_parenthesize(float probability) = 0;
  virtual CvQualifiers gen_cv_qualifiers(float const_prob,
                                         float volatile_prob) = 0;
};

class DefaultGeneratorRng : public GeneratorRng {
 public:
  explicit DefaultGeneratorRng(uint32_t seed) : rng_(seed) {}

  BinOp gen_bin_op(BinOpMask mask) override;
  UnOp gen_un_op(UnOpMask mask) override;
  ExprKind gen_expr_kind(const Weights& array) override;
  TypeKind gen_type_kind(const Weights& array) override;
  ScalarType gen_scalar_type() override;
  bool gen_boolean() override;
  uint64_t gen_u64(uint64_t min, uint64_t max) override;
  double gen_double(double min, double max) override;
  bool gen_parenthesize(float probability) override;
  CvQualifiers gen_cv_qualifiers(float const_prob,
                                 float volatile_prob) override;

 private:
  std::mt19937 rng_;
};

class ExprGenerator {
 public:
  ExprGenerator(std::unique_ptr<GeneratorRng> rng, const GenConfig& cfg)
      : rng_(std::move(rng)), cfg_(cfg) {}

  Expr generate();

 private:
  static constexpr char VAR[] = "x";

  Expr maybe_parenthesized(Expr expr);

  BooleanConstant gen_boolean_constant();
  IntegerConstant gen_integer_constant(const TypeConstraints& constraints);
  DoubleConstant gen_double_constant();
  VariableExpr gen_variable_expr(const TypeConstraints& constraints);
  BinaryExpr gen_binary_expr(const Weights& weights,
                             const TypeConstraints& constraints);
  UnaryExpr gen_unary_expr(const Weights& weights,
                           const TypeConstraints& constraints);
  TernaryExpr gen_ternary_expr(const Weights& weights,
                               const TypeConstraints& constraints);
  CastExpr gen_cast_expr(const Weights& weights,
                         const TypeConstraints& constraints);
  AddressOf gen_address_of_expr(const Weights& weights,
                                const TypeConstraints& constraints);
  MemberOf gen_member_of_expr(const Weights& weights,
                              const TypeConstraints& constraints);
  MemberOfPtr gen_member_of_ptr_expr(const Weights& weights,
                                     const TypeConstraints& constraints);
  ArrayIndex gen_array_index_expr(const Weights& weights,
                                  const TypeConstraints& constraints);

  Type gen_type(const Weights& weights, const TypeConstraints& constraints);
  QualifiedType gen_qualified_type(const Weights& weights,
                                   const TypeConstraints& constraints);
  PointerType gen_pointer_type(const Weights& weights,
                               const TypeConstraints& constraints);
  TaggedType gen_tagged_type(const TypeConstraints& constraints);
  ScalarType gen_scalar_type(const TypeConstraints& constraints);
  CvQualifiers gen_cv_qualifiers();

  Expr gen_with_weights(const Weights& weights,
                        const TypeConstraints& constraints);

 private:
  std::unique_ptr<GeneratorRng> rng_;
  GenConfig cfg_;
};

}  // namespace fuzzer

#endif  // INCLUDE_EXPR_GEN_H
