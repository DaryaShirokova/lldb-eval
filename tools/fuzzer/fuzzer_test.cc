#include <algorithm>
#include <cassert>
#include <cstddef>
#include <sstream>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tools/fuzzer/ast.h"
#include "tools/fuzzer/constraints.h"
#include "tools/fuzzer/expr_gen.h"

using namespace fuzzer;
using namespace testing;

namespace fuzzer {

void PrintTo(const Expr& expr, std::ostream* os) { *os << "`" << expr << "`"; }

}  // namespace fuzzer

class FakeGeneratorRng : public GeneratorRng {
 public:
  FakeGeneratorRng() {}

  BinOp gen_bin_op(BinOpMask) override {
    assert(!bin_ops_.empty());
    BinOp op = bin_ops_.back();
    bin_ops_.pop_back();

    return op;
  }

  UnOp gen_un_op(UnOpMask) override {
    assert(!un_ops_.empty());
    UnOp op = un_ops_.back();
    un_ops_.pop_back();

    return op;
  }

  ExprKind gen_expr_kind(const Weights&, const ExprKindMask&) override {
    assert(!expr_kinds_.empty());
    ExprKind kind = expr_kinds_.back();
    expr_kinds_.pop_back();

    return kind;
  }

  TypeKind gen_type_kind(const Weights&, const TypeKindMask&) override {
    if (type_kinds_.empty()) {
      return TypeKind::ScalarType;
    }

    TypeKind kind = type_kinds_.back();
    type_kinds_.pop_back();

    return kind;
  }

  ScalarType gen_scalar_type(EnumBitset<ScalarType>) override {
    if (scalar_types_.empty()) {
      return ScalarType::SignedInt;
    }
    ScalarType type = scalar_types_.back();
    scalar_types_.pop_back();

    return type;
  }

  bool gen_boolean() override {
    assert(!bools_.empty());
    bool constant = bools_.back();
    bools_.pop_back();

    return constant;
  }

  IntegerConstant gen_integer_constant(uint64_t, uint64_t) override {
    assert(!int_constants_.empty());
    IntegerConstant constant = int_constants_.back();
    int_constants_.pop_back();

    return constant;
  }

  DoubleConstant gen_double_constant(double, double) override {
    assert(!double_constants_.empty());
    DoubleConstant constant = double_constants_.back();
    double_constants_.pop_back();

    return constant;
  }

  bool gen_binop_ptr_expr(float) override { return false; }

  bool gen_binop_flip_operands(float) override { return false; }

  bool gen_binop_ptrdiff_expr(float) override { return false; }

  CvQualifiers gen_cv_qualifiers(float, float) override {
    assert(!cv_qualifiers_.empty());
    CvQualifiers cv = cv_qualifiers_.back();
    cv_qualifiers_.pop_back();

    return cv;
  }

  bool gen_parenthesize(float) override { return false; }

  static FakeGeneratorRng from_expr(const Expr& expr) {
    FakeGeneratorRng rng;
    std::visit(rng, expr);

    std::reverse(rng.un_ops_.begin(), rng.un_ops_.end());
    std::reverse(rng.bin_ops_.begin(), rng.bin_ops_.end());
    std::reverse(rng.bools_.begin(), rng.bools_.end());
    std::reverse(rng.int_constants_.begin(), rng.int_constants_.end());
    std::reverse(rng.double_constants_.begin(), rng.double_constants_.end());
    std::reverse(rng.expr_kinds_.begin(), rng.expr_kinds_.end());
    std::reverse(rng.cv_qualifiers_.begin(), rng.cv_qualifiers_.end());

    return rng;
  }

  void operator()(const UnaryExpr& e) {
    expr_kinds_.push_back(ExprKind::UnaryExpr);
    un_ops_.push_back(e.op());
    std::visit(*this, e.expr());
  }

  void operator()(const BinaryExpr& e) {
    expr_kinds_.push_back(ExprKind::BinaryExpr);
    bin_ops_.push_back(e.op());
    std::visit(*this, e.lhs());
    std::visit(*this, e.rhs());
  }

  void operator()(const VariableExpr&) {
    expr_kinds_.push_back(ExprKind::VariableExpr);
  }

  void operator()(const IntegerConstant& e) {
    expr_kinds_.push_back(ExprKind::IntegerConstant);
    int_constants_.push_back(e);
  }

  void operator()(const DoubleConstant& e) {
    expr_kinds_.push_back(ExprKind::DoubleConstant);
    double_constants_.push_back(e);
  }

  void operator()(const BooleanConstant& e) {
    expr_kinds_.push_back(ExprKind::BooleanConstant);
    bools_.push_back(e.value());
  }

  void operator()(const ParenthesizedExpr& e) { std::visit(*this, e.expr()); }

  void operator()(const AddressOf& e) {
    expr_kinds_.push_back(ExprKind::AddressOf);
    std::visit(*this, e.expr());
  }

  void operator()(const MemberOf& e) {
    expr_kinds_.push_back(ExprKind::MemberOf);
    std::visit(*this, e.expr());
  }

  void operator()(const MemberOfPtr& e) {
    expr_kinds_.push_back(ExprKind::MemberOfPtr);
    std::visit(*this, e.expr());
  }

  void operator()(const ArrayIndex& e) {
    expr_kinds_.push_back(ExprKind::ArrayIndex);
    std::visit(*this, e.expr());
    std::visit(*this, e.idx());
  }

  void operator()(const TernaryExpr& e) {
    expr_kinds_.push_back(ExprKind::TernaryExpr);
    std::visit(*this, e.cond());
    std::visit(*this, e.lhs());
    std::visit(*this, e.rhs());
  }

  void operator()(const CastExpr& e) {
    expr_kinds_.push_back(ExprKind::CastExpr);
    std::visit(*this, e.type());
    std::visit(*this, e.expr());
  }

  void operator()(const DereferenceExpr& e) {
    expr_kinds_.push_back(ExprKind::DereferenceExpr);
    std::visit(*this, e.expr());
  }

  void operator()(const QualifiedType& e) {
    cv_qualifiers_.push_back(e.cv_qualifiers());
    std::visit(*this, e.type());
  }

  void operator()(const PointerType& e) {
    type_kinds_.push_back(TypeKind::PointerType);
    (*this)(e.type());
  }

  void operator()(const TaggedType& e) {
    type_kinds_.push_back(TypeKind::TaggedType);
    tagged_types_.push_back(e.name());
  }

  void operator()(const ScalarType& e) {
    type_kinds_.push_back(TypeKind::ScalarType);
    scalar_types_.push_back(e);
  }

 private:
  std::vector<UnOp> un_ops_;
  std::vector<BinOp> bin_ops_;
  std::vector<IntegerConstant> int_constants_;
  std::vector<DoubleConstant> double_constants_;
  std::vector<bool> bools_;
  std::vector<ExprKind> expr_kinds_;
  std::vector<TypeKind> type_kinds_;
  std::vector<CvQualifiers> cv_qualifiers_;
  std::vector<ScalarType> scalar_types_;
  std::vector<std::string> tagged_types_;
};

struct Mismatch {
  std::string lhs;
  std::string rhs;

  Mismatch(std::string lhs, std::string rhs) : lhs(lhs), rhs(rhs) {}
};

class AstComparator {
 public:
  void operator()(const UnaryExpr& lhs, const UnaryExpr& rhs) {
    if (lhs.op() != rhs.op()) {
      add_mismatch(lhs, rhs);
      return;
    }

    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const BinaryExpr& lhs, const BinaryExpr& rhs) {
    if (lhs.op() != rhs.op()) {
      add_mismatch(lhs, rhs);
      return;
    }

    std::visit(*this, lhs.lhs(), rhs.lhs());
    std::visit(*this, lhs.rhs(), rhs.rhs());
  }

  void operator()(const VariableExpr& lhs, const VariableExpr& rhs) {
    if (lhs.name() != rhs.name()) {
      add_mismatch(lhs, rhs);
    }
  }

  void operator()(const IntegerConstant& lhs, const IntegerConstant& rhs) {
    if (lhs.value() != rhs.value()) {
      add_mismatch(lhs, rhs);
    }
  }

  void operator()(const DoubleConstant& lhs, const DoubleConstant& rhs) {
    if (lhs.value() != rhs.value()) {
      add_mismatch(lhs, rhs);
    }
  }

  void operator()(const DereferenceExpr& lhs, const DereferenceExpr& rhs) {
    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const ParenthesizedExpr& lhs, const ParenthesizedExpr& rhs) {
    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const AddressOf& lhs, const AddressOf& rhs) {
    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const MemberOf& lhs, const MemberOf& rhs) {
    if (lhs.field() != rhs.field()) {
      add_mismatch(lhs, rhs);
      return;
    }

    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const MemberOfPtr& lhs, const MemberOfPtr& rhs) {
    if (lhs.field() != rhs.field()) {
      add_mismatch(lhs, rhs);
      return;
    }

    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const ArrayIndex& lhs, const ArrayIndex& rhs) {
    std::visit(*this, lhs.expr(), rhs.expr());
    std::visit(*this, lhs.idx(), rhs.idx());
  }

  void operator()(const TernaryExpr& lhs, const TernaryExpr& rhs) {
    std::visit(*this, lhs.cond(), rhs.cond());
    std::visit(*this, lhs.lhs(), rhs.lhs());
    std::visit(*this, lhs.rhs(), rhs.rhs());
  }

  void operator()(const CastExpr& lhs, const CastExpr& rhs) {
    std::visit(*this, lhs.type(), rhs.type());
    std::visit(*this, lhs.expr(), rhs.expr());
  }

  void operator()(const QualifiedType& lhs, const QualifiedType& rhs) {
    std::visit(*this, lhs.type(), rhs.type());
    if (lhs.cv_qualifiers() != rhs.cv_qualifiers()) {
      add_mismatch(lhs.cv_qualifiers(), rhs.cv_qualifiers());
    }
  }

  void operator()(const PointerType& lhs, const PointerType& rhs) {
    (*this)(lhs.type(), rhs.type());
  }

  void operator()(const TaggedType& lhs, const TaggedType& rhs) {
    if (lhs.name() != rhs.name()) {
      add_mismatch(lhs.name(), rhs.name());
    }
  }

  void operator()(ScalarType lhs, ScalarType rhs) {
    if (lhs != rhs) {
      add_mismatch(lhs, rhs);
    }
  }

  void operator()(BooleanConstant lhs, BooleanConstant rhs) {
    if (lhs.value() != rhs.value()) {
      add_mismatch(lhs.value(), rhs.value());
    }
  }

  template <typename T, typename U,
            typename = std::enable_if_t<!std::is_same_v<T, U>>>
  void operator()(const T& lhs, const U& rhs) {
    add_mismatch(lhs, rhs);
  }

  const std::vector<Mismatch>& mismatches() const { return mismatches_; }

 private:
  template <typename T, typename U>
  void add_mismatch(const T& lhs, const U& rhs) {
    std::ostringstream lhs_stream;
    std::ostringstream rhs_stream;

    lhs_stream << lhs;
    rhs_stream << rhs;

    mismatches_.emplace_back(lhs_stream.str(), rhs_stream.str());
  }

 private:
  std::vector<Mismatch> mismatches_;
};

MATCHER_P(MatchesAst, expected,
          (negation ? "does not match AST " : "matches AST ") +
              PrintToString(expected.get())) {
  AstComparator cmp;

  std::visit(cmp, expected.get(), arg);
  const auto& mismatches = cmp.mismatches();
  if (mismatches.empty()) {
    return true;
  }

  *result_listener << "with mismatches occuring as follows:\n";
  for (const auto& e : mismatches) {
    *result_listener << "* Expected: " << PrintToString(e.lhs) << "\n"
                     << "*   Actual: " << PrintToString(e.rhs) << "\n";
  }

  return false;
}

struct PrecedenceTestParam {
  std::string str;
  std::shared_ptr<Expr> expr;

  PrecedenceTestParam(std::string str, Expr expr)
      : str(std::move(str)), expr(std::make_shared<Expr>(std::move(expr))) {}
};

std::ostream& operator<<(std::ostream& os, const PrecedenceTestParam& param) {
  return os << "`" << param.str << "`";
}

class OperatorPrecedence : public TestWithParam<PrecedenceTestParam> {};

TEST_P(OperatorPrecedence, CorrectAst) {
  const auto& param = GetParam();

  auto fake_rng = std::make_unique<FakeGeneratorRng>(
      FakeGeneratorRng::from_expr(*param.expr));

  ExprGenerator gen(std::move(fake_rng), GenConfig());
  auto maybe_expr = gen.generate();
  auto& expr = maybe_expr.value();
  std::ostringstream os;
  os << expr;

  EXPECT_THAT(expr, MatchesAst(std::cref(*param.expr)));
  EXPECT_THAT(os.str(), StrEq(param.str));
}

std::vector<PrecedenceTestParam> gen_precedence_params() {
  std::vector<PrecedenceTestParam> params;
  {
    // clang-format off
    Expr expected = BinaryExpr(
        IntegerConstant(3),
        BinOp::Mult,
        ParenthesizedExpr(
            BinaryExpr(IntegerConstant(4), BinOp::Plus, IntegerConstant(5))));
    // clang-format on

    std::string str = "3 * (4 + 5)";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = BinaryExpr(
        BinaryExpr(IntegerConstant(3), BinOp::Mult, IntegerConstant(4)),
        BinOp::Plus,
        IntegerConstant(5));
    // clang-format on

    std::string str = "3 * 4 + 5";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = BinaryExpr(
        BinaryExpr(IntegerConstant(3), BinOp::Minus, IntegerConstant(4)),
        BinOp::Plus,
        IntegerConstant(5));
    // clang-format on

    std::string str = "3 - 4 + 5";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = BinaryExpr(
        IntegerConstant(3),
        BinOp::Minus,
        ParenthesizedExpr(
            BinaryExpr(IntegerConstant(4), BinOp::Plus, IntegerConstant(5))));
    // clang-format on

    std::string str = "3 - (4 + 5)";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = UnaryExpr(UnOp::Neg,
                              UnaryExpr(UnOp::Neg, IntegerConstant(1)));
    // clang-format on

    std::string str = "- -1";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected =
        CastExpr(ScalarType::SignedInt, IntegerConstant(50));
    // clang-format on

    std::string str = "(int) 50";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = BinaryExpr(
        CastExpr(ScalarType::SignedInt, IntegerConstant(50)),
        BinOp::Plus,
        IntegerConstant(1));
    // clang-format on

    std::string str = "(int) 50 + 1";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = CastExpr(ScalarType::SignedInt,
        ParenthesizedExpr(
            BinaryExpr(IntegerConstant(50), BinOp::Plus, IntegerConstant(1))));
    // clang-format on

    std::string str = "(int) (50 + 1)";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = UnaryExpr(
        UnOp::BitNot,
        ParenthesizedExpr(TernaryExpr(
            IntegerConstant(1),
            IntegerConstant(2),
            IntegerConstant(3))));
    // clang-format on

    std::string str = "~(1 ? 2 : 3)";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = BinaryExpr(
        IntegerConstant(1),
        BinOp::Mult,
        ParenthesizedExpr(
            TernaryExpr(
                BooleanConstant(true),
                IntegerConstant(0),
                TernaryExpr(
                    BooleanConstant(false),
                    IntegerConstant(1),
                    IntegerConstant(2)))));
    // clang-format on

    std::string str = "1 * (true ? 0 : false ? 1 : 2)";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    // clang-format off
    Expr expected = BinaryExpr(
        IntegerConstant(4),
        BinOp::Mult,
        ParenthesizedExpr(
            TernaryExpr(ParenthesizedExpr(TernaryExpr(
                BinaryExpr(IntegerConstant(1), BinOp::Eq, IntegerConstant(2)),
                BooleanConstant(false),
                BooleanConstant(true))),
        IntegerConstant(1),
        IntegerConstant(0))));
    // clang-format on

    std::string str = "4 * ((1 == 2 ? false : true) ? 1 : 0)";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    Expr expected = UnaryExpr(UnOp::Neg, DoubleConstant(1.5));

    std::string str = "-1.5";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    Expr expected = IntegerConstant(0xbadf00d, IntegerConstant::Base::Hex,
                                    IntegerConstant::Length::LongLong,
                                    IntegerConstant::Signedness::Unsigned);

    std::string str = "0xbadf00dLLU";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    Expr expected = IntegerConstant(0b1001'1001, IntegerConstant::Base::Bin,
                                    IntegerConstant::Length::Long,
                                    IntegerConstant::Signedness::Signed);

    std::string str = "0b10011001L";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    Expr expected = DoubleConstant(1.5, DoubleConstant::Format::Default,
                                   DoubleConstant::Length::Float);

    std::string str = "1.5f";
    params.emplace_back(std::move(str), std::move(expected));
  }
  {
    Expr expected = DoubleConstant(0x0.1p-1070, DoubleConstant::Format::Hex,
                                   DoubleConstant::Length::Double);

#if defined(_MSC_VER) && _MSC_VER < 1928
    // Apparently MSVC seems to not be adhering to the C++11 standard properly
    // and it takes into account the precision modifier when printing hex
    // floats (https://en.cppreference.com/w/cpp/locale/num_put/put#Notes).
    //
    // Hence we're using this ifdef guard as a temporary workaround.
    std::string str = "0x0.000000p-1022";
#else
    std::string str = "0x0.0000000000001p-1022";
#endif

    params.emplace_back(std::move(str), std::move(expected));
  }

  return params;
}

INSTANTIATE_TEST_SUITE_P(AstGen, OperatorPrecedence,
                         ValuesIn(gen_precedence_params()));

struct TypePrintTestParam {
  std::string str;
  std::shared_ptr<Type> type;

  TypePrintTestParam(std::string str, Type type)
      : str(std::move(str)), type(std::make_shared<Type>(std::move(type))) {}
};

std::ostream& operator<<(std::ostream& os, const TypePrintTestParam& param) {
  return os << param.str;
}

class TypePrinting : public TestWithParam<TypePrintTestParam> {};

TEST_P(TypePrinting, CorrectStrType) {
  const auto& param = GetParam();

  std::ostringstream os;
  os << *param.type;
  EXPECT_THAT(os.str(), StrEq(param.str));
}

std::vector<TypePrintTestParam> gen_typing_params() {
  std::vector<TypePrintTestParam> params;

  {
    auto type = ScalarType::SignedInt;
    std::string str = "int";
    params.emplace_back(std::move(str), std::move(type));
  }

  {
    Type type(PointerType(QualifiedType(
        PointerType(QualifiedType(ScalarType::Char, CvQualifier::Const)))));
    std::string str = "const char**";

    params.emplace_back(std::move(str), std::move(type));
  }

  {
    PointerType type(QualifiedType(
        PointerType(QualifiedType(ScalarType::SignedInt, CvQualifier::Const)),
        CvQualifier::Volatile));
    std::string str = "const int* volatile*";

    params.emplace_back(std::move(str), std::move(type));
  }

  {
    PointerType type(
        QualifiedType(TaggedType("TestStruct"), CvQualifier::Const));
    std::string str = "const TestStruct*";

    params.emplace_back(std::move(str), std::move(type));
  }

  {
    PointerType type(QualifiedType(PointerType(QualifiedType(ScalarType::Void)),
                                   CvQualifier::Const));
    std::string str = "void* const*";

    params.emplace_back(std::move(str), std::move(type));
  }

  return params;
}

INSTANTIATE_TEST_SUITE_P(AstGen, TypePrinting, ValuesIn(gen_typing_params()));

TEST(Constraints, ScalarValues) {
  SpecificTypes float_only = FLOAT_TYPES;
  SpecificTypes int_only = INT_TYPES;

  TypeConstraints float_constraints = float_only;
  TypeConstraints int_constraints = int_only;

  TypeConstraints any = AnyType();
  TypeConstraints none;

  PointerType void_ptr_type{QualifiedType(ScalarType::Void)};

  EXPECT_THAT(float_only.allows_any_of(ScalarType::Float), IsTrue());
  EXPECT_THAT(float_only.allows_any_of(ScalarType::Double), IsTrue());
  EXPECT_THAT(float_only.allows_any_of(ScalarType::LongDouble), IsTrue());

  EXPECT_THAT(float_only.allows_any_of(ScalarType::Void), IsFalse());
  EXPECT_THAT(float_only.allows_any_of(ScalarType::Bool), IsFalse());
  EXPECT_THAT(float_only.allows_any_of(ScalarType::SignedInt), IsFalse());
  EXPECT_THAT(float_only.allows_any_of(ScalarType::UnsignedLong), IsFalse());

  EXPECT_THAT(int_only.allowed_tagged_types(), IsEmpty());
  EXPECT_THAT(float_only.allowed_tagged_types(), IsEmpty());

  EXPECT_THAT(int_only.allows_non_void_pointer(), IsFalse());
  EXPECT_THAT(float_only.allows_non_void_pointer(), IsFalse());

  EXPECT_THAT(int_only.allows_void_pointer(), IsFalse());
  EXPECT_THAT(float_only.allows_void_pointer(), IsFalse());

  EXPECT_THAT(int_only.allows_any_of(ScalarType::Float), IsFalse());
  EXPECT_THAT(int_only.allows_any_of(ScalarType::Double), IsFalse());
  EXPECT_THAT(int_only.allows_any_of(ScalarType::LongDouble), IsFalse());

  EXPECT_THAT(int_only.allows_any_of(ScalarType::Void), IsFalse());
  EXPECT_THAT(int_only.allows_any_of(ScalarType::Bool), IsTrue());
  EXPECT_THAT(int_only.allows_any_of(ScalarType::SignedInt), IsTrue());
  EXPECT_THAT(int_only.allows_any_of(ScalarType::UnsignedLong), IsTrue());

  EXPECT_THAT(float_constraints.allows_type(ScalarType::Float), IsTrue());
  EXPECT_THAT(float_constraints.allows_type(ScalarType::SignedInt), IsFalse());
  EXPECT_THAT(float_constraints.allows_type(TaggedType("Test")), IsFalse());
  EXPECT_THAT(float_constraints.allows_type(void_ptr_type), IsFalse());

  EXPECT_THAT(int_constraints.allows_type(ScalarType::Float), IsFalse());
  EXPECT_THAT(int_constraints.allows_type(ScalarType::SignedInt), IsTrue());
  EXPECT_THAT(int_constraints.allows_type(TaggedType("Test")), IsFalse());
  EXPECT_THAT(int_constraints.allows_type(void_ptr_type), IsFalse());

  EXPECT_THAT(any.allows_type(ScalarType::Float), IsTrue());
  EXPECT_THAT(any.allows_type(ScalarType::SignedInt), IsTrue());

  EXPECT_THAT(none.allows_type(ScalarType::Float), IsFalse());
  EXPECT_THAT(none.allows_type(ScalarType::SignedInt), IsFalse());
}

TEST(Constraints, TaggedTypes) {
  SpecificTypes test_struct =
      std::unordered_set<TaggedType>{{TaggedType("TestStruct")}};

  EXPECT_THAT(test_struct.allows_any_of(ScalarMask::all_set()), IsFalse());
  EXPECT_THAT(test_struct.allowed_tagged_types(),
              UnorderedElementsAre(TaggedType("TestStruct")));
  EXPECT_THAT(test_struct.allows_non_void_pointer(), IsFalse());
  EXPECT_THAT(test_struct.allows_void_pointer(), IsFalse());

  TypeConstraints any = AnyType();
  TypeConstraints none;

  EXPECT_THAT(any.allows_tagged_types(), IsTrue());
  EXPECT_THAT(any.allowed_tagged_types(), IsNull());
  EXPECT_THAT(any.allows_type(TaggedType("TestStruct")), IsTrue());

  EXPECT_THAT(none.allows_tagged_types(), IsFalse());
  EXPECT_THAT(none.allowed_tagged_types(), IsNull());
  EXPECT_THAT(none.allows_type(TaggedType("TestStruct")), IsFalse());
}

TEST(Constraints, PointerTypes) {
  SpecificTypes int_ptr = SpecificTypes::make_pointer_constraints(
      SpecificTypes(ScalarMask{ScalarType::SignedInt}));
  SpecificTypes void_ptr = SpecificTypes::make_pointer_constraints(
      SpecificTypes(), VoidPointerConstraint::Allow);

  EXPECT_THAT(int_ptr.allows_any_of(ScalarMask::all_set()), IsFalse());
  EXPECT_THAT(int_ptr.allowed_tagged_types(), IsEmpty());
  EXPECT_THAT(int_ptr.allows_non_void_pointer(), IsTrue());
  EXPECT_THAT(int_ptr.allows_void_pointer(), IsFalse());

  EXPECT_THAT(void_ptr.allows_any_of(ScalarMask::all_set()), IsFalse());
  EXPECT_THAT(void_ptr.allowed_tagged_types(), IsEmpty());
  EXPECT_THAT(void_ptr.allows_non_void_pointer(), IsFalse());
  EXPECT_THAT(void_ptr.allows_void_pointer(), IsTrue());

  PointerType const_int_ptr{
      QualifiedType(ScalarType::SignedInt, CvQualifier::Const)};
  PointerType volatile_void_ptr{
      QualifiedType(ScalarType::Void, CvQualifier::Volatile)};

  TypeConstraints int_ptr_constraints = int_ptr;
  TypeConstraints void_ptr_constraints = void_ptr;

  TypeConstraints int_constraints = int_ptr.allowed_to_point_to();
  TypeConstraints void_constraints = void_ptr.allowed_to_point_to();

  EXPECT_THAT(int_ptr_constraints.allows_type(const_int_ptr), IsTrue());
  EXPECT_THAT(int_ptr_constraints.allows_type(volatile_void_ptr), IsFalse());
  EXPECT_THAT(int_constraints.allows_any_of(ScalarType::SignedInt), IsTrue());
  EXPECT_THAT(int_constraints.allows_any_of(ScalarType::Void), IsFalse());

  EXPECT_THAT(void_ptr_constraints.allows_type(const_int_ptr), IsFalse());
  EXPECT_THAT(void_ptr_constraints.allows_type(volatile_void_ptr), IsTrue());
  EXPECT_THAT(void_constraints.allows_any_of(ScalarType::SignedInt), IsFalse());

  // Due to the way we represent constraints, we cannot state that we support
  // void types :(
  EXPECT_THAT(void_constraints.allows_any_of(ScalarType::Void), IsFalse());

  TypeConstraints any = AnyType();
  TypeConstraints none;

  EXPECT_THAT(any.allows_type(const_int_ptr), IsTrue());
  EXPECT_THAT(any.allows_type(volatile_void_ptr), IsTrue());

  EXPECT_THAT(none.allows_type(const_int_ptr), IsFalse());
  EXPECT_THAT(none.allows_type(volatile_void_ptr), IsFalse());
}

TEST(Constraints, Unsatisfiability) {
  TypeConstraints default_ctor;
  TypeConstraints default_specific_types = SpecificTypes();

  EXPECT_THAT(default_ctor.satisfiable(), IsFalse());
  EXPECT_THAT(default_specific_types.satisfiable(), IsFalse());
}
