/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.comet.serde

import org.apache.spark.sql.{CometTestBase, Row}
import org.apache.spark.sql.catalyst.expressions.{Add, Alias, Ascending, AttributeReference, BloomFilterMightContain, BoundReference, CaseWhen, Cast, Coalesce, ConcatWs, CreateArray, CreateNamedStruct, DateAdd, DateDiff, DayOfWeek, EqualNullSafe, EqualTo, EvalMode, Expression, FromUTCTimestamp, GetStructField, GreaterThan, GreaterThanOrEqual, Hour, If, In, IsNotNull, LessThan, Literal, Minute, Murmur3Hash, NamedExpression, Not, OctetLength, PreciseTimestampConversion, Rand, Round, Second, SecondsToTimestamp, SortOrder, StringLPad, StringRepeat, StringRPad, ToUTCTimestamp, TruncDate, TruncTimestamp, UnixDate, UnixTimestamp, WeekDay, XxHash64}
import org.apache.spark.sql.execution.{FilterExec, ProjectExec, ScalarSubquery, SubqueryExec}
import org.apache.spark.sql.types._

import org.apache.comet.CometConf
import org.apache.comet.CometSparkSessionExtensions.{isSpark40Plus, isSpark41Plus}
import org.apache.comet.shims.CometTypeShim

/** Direct type-gate tests avoid optimizer folding and exercise types Parquet cannot store. */
class CometScalarSubquerySuite extends CometTestBase with CometTypeShim {

  private def struct(dt: DataType): StructType = StructType(Seq(StructField("value", dt)))

  private lazy val emptyInput = spark.range(0).queryExecution.sparkPlan

  private def subquery(dt: DataType): ScalarSubquery = {
    // Inspect the declared result type without executing or optimizing the typed NULL away.
    val plan = ProjectExec(Seq(Alias(Literal.create(null, dt), "result")()), emptyInput)
    ScalarSubquery(SubqueryExec("type-check", plan), NamedExpression.newExprId)
  }

  private def supported(dt: DataType): Boolean =
    CometScalarSubquery.getSupportLevel(subquery(dt)) == Compatible()

  test("array hash fusion preserves argument order and evaluation boundaries") {
    val value = AttributeReference("value", LongType, nullable = true)()
    val children = Seq(value, Literal(1L), Literal.create(null, LongType))
    val array = CreateArray(children)
    assert(
      HashUtils.hashChildren(Seq(Literal(7L), array, value)) ==
        Seq(Literal(7L)) ++ children ++ Seq(value))
    val proto =
      CometMurmur3Hash.convert(Murmur3Hash(Seq(array), 17), Seq(value), binding = false).get
    assert(proto.getScalarFunc.getArgsCount == 4)
    assert(proto.getScalarFunc.getArgs(3).getLiteral.getIntVal == 17)
    val xx = CometXxHash64.convert(XxHash64(Seq(array), 17L), Seq(value), binding = false).get
    assert(xx.getScalarFunc.getArgsCount == 4)
    assert(xx.getScalarFunc.getArgs(3).getLiteral.getLongVal == 17L)
    for (dataType <- Seq(
        BooleanType,
        FloatType,
        DoubleType,
        DecimalType(18, 2),
        StringType,
        BinaryType,
        struct(LongType),
        ArrayType(LongType),
        MapType(StringType, LongType))) {
      val value = AttributeReference("value", dataType, nullable = true)()
      val values = Seq(value, Literal.create(null, dataType))
      assert(HashUtils.hashChildren(Seq(CreateArray(values))) == values)
    }
    val fallible = array.copy(children = Seq(Add(value, Literal(1L), EvalMode.ANSI)))
    val stateful = array.copy(children = Seq(Cast(Rand(42L), LongType)))
    val nullable = Literal.create(null, ArrayType(LongType))
    val conditional = If(Literal(true), array, nullable)
    val empty = CreateArray(Seq.empty)
    for (child <- Seq(fallible, stateful, nullable, conditional, empty)) {
      assert(HashUtils.hashChildren(Seq(child)) == Seq(child))
    }
  }

  test("deferred TopK admits pure integer keys only in its comparator scope") {
    val id = AttributeReference("id", IntegerType)()
    val value = GetStructField(subquery(struct(IntegerType)), 0)
    val add = Add(id, value, EvalMode.ANSI)
    val order = SortOrder(add, Ascending)
    assert(CometScalarSubquery.usesDeferredTopK(Seq(order), 7, Seq(id)) == isSpark40Plus)
    assert(!CometScalarSubquery.usesDeferredTopK(Seq(order), 16, Seq(id)))
    assert(!CometScalarSubquery.usesDeferredTopK(Seq(order), 7, Seq(Alias(add, "result")())))
    val nested = SortOrder(Add(add, value, EvalMode.ANSI), Ascending)
    assert(!CometScalarSubquery.usesDeferredTopK(Seq(nested), 7, Seq(id)))
    assert(QueryPlanSerde.operatorExprToProto(order, Seq(id)).isEmpty)
    CometScalarSubquery.withDeferredTopK(enabled = true) {
      val proto = QueryPlanSerde.operatorExprToProto(order, Seq(id))
      assert(proto.exists(_.getSortOrder.getChild.hasAdd))
      assert(QueryPlanSerde.operatorExprToProto(nested, Seq(id)).isEmpty)
    }
    assert(QueryPlanSerde.operatorExprToProto(order, Seq(id)).isEmpty)
  }

  test("struct context preserves row errors independently of width and NULL masks") {
    val id = AttributeReference("id", LongType, nullable = false)()
    val text = AttributeReference("text", StringType)()
    val scalar = subquery(struct(LongType))
    val value = GetStructField(scalar, 0)
    val total = If(EqualTo(id, Literal(0L)), value, Literal(1L))
    val cast = Cast(text, LongType, None, EvalMode.ANSI)
    val fallible = If(EqualTo(id, Literal(0L)), value, cast)
    val plan = ProjectExec(Seq(Alias(value, "scalar")(), Alias(id, "id")()), emptyInput)
    CometScalarSubquery.tagPlanInputs(plan)
    CometScalarSubquery.withPlanInputs(plan) {
      assert(CometScalarSubquery.consumerFallbackReason(total).isEmpty)
      assert(CometScalarSubquery.consumerFallbackReason(cast).isDefined)
      assert(CometScalarSubquery.consumerFallbackReason(fallible).isDefined)
      assert(CometScalarSubquery.broadcastBuildKeysFallbackReason(Seq(total)).isEmpty)
      assert(CometScalarSubquery.broadcastBuildKeysFallbackReason(Seq(cast)).isDefined)
      val scalarText = GetStructField(subquery(struct(StringType)), 0)
      val concat = ConcatWs(Seq(Literal.create(null, StringType), scalarText))
      assert(CometScalarSubquery.consumerFallbackReason(concat).isEmpty)
      val parsed = Cast(Cast(scalarText, IntegerType, None, EvalMode.ANSI), StringType)
      assert(
        CometScalarSubquery
          .consumerFallbackReason(concat.copy(children = Seq(concat.children.head, parsed)))
          .isDefined)
      val pad = StringLPad(Literal("x"), Literal(8), Literal("0"))
      assert(CometScalarSubquery.consumerFallbackReason(pad).isEmpty)
      val unknownLength = pad.copy(len = AttributeReference("length", IntegerType)())
      assert(CometScalarSubquery.consumerFallbackReason(unknownLength).isDefined)
      val large = StringRepeat(Literal("xx"), Literal(Int.MaxValue))
      assert(CometScalarSubquery.consumerFallbackReason(large).isDefined)
      val wide = StringLPad(text, Literal(262144), Literal("x"))
      assert(
        CometScalarSubquery
          .projectionConsumerFallbackReason(Seq(value, wide), Seq.empty)
          .isDefined)
      assert(CometScalarSubquery.operatorArgumentFallbackReason(wide).isDefined)
      val order = SortOrder(text, Ascending)
      assert(CometScalarSubquery.operatorArgumentFallbackReason(order).isEmpty)
      assert(CometScalarSubquery.consumerFallbackReason(order).isEmpty)
      assert(
        CometScalarSubquery.operatorArgumentFallbackReason(SortOrder(wide, Ascending)).isDefined)
      assert(CometScalarSubquery.consumerFallbackReason(SortOrder(large, Ascending)).isDefined)
      assert(
        CometScalarSubquery.projectionConsumerFallbackReason(Seq(value, text), Seq.empty).isEmpty)
      val selected =
        If(EqualTo(id, Literal(0L)), GetStructField(subquery(struct(StringType)), 0), text)
      val bounded = StringLPad(selected, Literal(8), Literal("0"))
      assert(CometScalarSubquery.consumerFallbackReason(bounded).isEmpty)
      assert(
        CometScalarSubquery.projectionConsumerFallbackReason(Seq(bounded), Seq.empty).isEmpty)
      val larger = bounded.copy(len = Literal(256))
      assert(CometScalarSubquery.consumerFallbackReason(larger).isEmpty)
      assert(CometScalarSubquery.projectionConsumerFallbackReason(Seq(larger), Seq.empty).isEmpty)
      val fixed = StringLPad(
        If(EqualTo(id, Literal(0L)), Literal("x"), Literal("y")),
        Literal(8),
        Literal("0"))
      assert(
        CometScalarSubquery
          .projectionConsumerFallbackReason(Seq(value, fixed), Seq.empty)
          .isEmpty)
    }
    assert(CometScalarSubquery.consumerFallbackReason(cast).isEmpty)
    assert(CometScalarSubquery.broadcastBuildKeysFallbackReason(Seq(cast)).isEmpty)
  }

  test("conditional and broadcast Bloom initialization retain their own stage gates") {
    val bloom =
      BloomFilterMightContain(GetStructField(subquery(struct(BinaryType)), 0), Literal(1L))
    val conditionals = Seq(
      If(Literal(true), Literal(true), bloom),
      CaseWhen(Seq(Literal(true) -> Literal(true)), Some(bloom)),
      Coalesce(Seq(Literal(true), bloom)))
    Seq("CODEGEN_ONLY", "NO_CODEGEN").foreach { mode =>
      withSQLConf("spark.sql.codegen.factoryMode" -> mode) {
        conditionals.foreach { conditional =>
          assert(
            CometScalarSubquery.consumerFallbackReason(conditional).isDefined ==
              (mode == "NO_CODEGEN"),
            conditional)
        }
        val safe = Coalesce(Seq(GetStructField(subquery(struct(BooleanType)), 0), Literal(true)))
        assert(CometScalarSubquery.consumerFallbackReason(safe).isEmpty)
        assert(QueryPlanSerde.exprToProtoInternal(safe, Seq.empty, binding = false).isDefined)
        assert(CometScalarSubquery.broadcastBuildKeysFallbackReason(Seq(bloom)).isDefined)
        val text = If(bloom, Literal("x"), Literal("y"))
        val concat = ConcatWs(Seq(Literal.create(null, StringType), text))
        assert(CometScalarSubquery.consumerFallbackReason(concat).isDefined)
        val round = Round(Cast(bloom, LongType), Literal.create(null, IntegerType))
        assert(
          CometScalarSubquery.consumerFallbackReason(round).isDefined == (mode != "NO_CODEGEN"))
        assert(CometScalarSubquery.consumerFallbackReason(round.copy(scale = Literal(0))).isEmpty)
        val parsed = Cast(GetStructField(subquery(struct(StringType)), 0), LongType)
        assert(CometScalarSubquery.consumerFallbackReason(round.copy(child = parsed)).isEmpty)
      }
    }
  }

  test("native NULL masks cannot guard interpreted Bloom initialization") {
    val bloom =
      BloomFilterMightContain(GetStructField(subquery(struct(BinaryType)), 0), Literal(1L))
    val nullable = Literal.create(null, BooleanType)
    val consumers = Seq(
      EqualTo(nullable, bloom),
      LessThan(nullable, bloom),
      GreaterThan(nullable, bloom),
      GreaterThanOrEqual(nullable, bloom))
    for (mode <- Seq("NO_CODEGEN", "CODEGEN_ONLY")) {
      withSQLConf("spark.sql.codegen.factoryMode" -> mode) {
        consumers.foreach { consumer =>
          QueryPlanSerde.withSharedSubexpressions(Seq(consumer)) {
            assert(
              CometScalarSubquery.consumerFallbackReason(consumer).isDefined ==
                (mode == "NO_CODEGEN"),
              consumer)
          }
        }
        QueryPlanSerde.withSharedSubexpressions(Seq.empty) {
          assert(
            CometScalarSubquery.consumerFallbackReason(EqualTo(Literal(true), bloom)).isEmpty)
          val safe = GetStructField(subquery(struct(BooleanType)), 0)
          assert(CometScalarSubquery.consumerFallbackReason(EqualTo(nullable, safe)).isEmpty)
        }
      }
    }
  }

  private def versionSpecificTypes: Seq[DataType] = {
    val strings =
      if (isSpark40Plus) Seq(DataType.fromDDL("STRING COLLATE UTF8_LCASE")) else Seq.empty
    val times = if (isSpark41Plus) Seq(DataType.fromDDL("TIME")) else Seq.empty
    strings ++ times ++ variantType.toSeq
  }

  private val scalarTypes: Seq[DataType] = Seq(
    BooleanType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    StringType,
    BinaryType,
    DateType,
    TimestampType,
    TimestampNTZType,
    NullType,
    DecimalType(1, 0),
    DecimalType(10, 2),
    DecimalType(38, 38))

  private def nestedTypes(dt: DataType): Seq[DataType] = Seq(
    dt,
    struct(dt),
    struct(struct(dt)),
    ArrayType(dt),
    MapType(dt, IntegerType),
    MapType(IntegerType, dt),
    struct(ArrayType(MapType(StringType, dt))))

  test("non-struct scalar subqueries use the shared type gate defaults") {
    val types = scalarTypes ++ versionSpecificTypes ++ Seq(
      CalendarIntervalType,
      YearMonthIntervalType(),
      DayTimeIntervalType(),
      CharType(5),
      VarcharType(5),
      StructType(Nil),
      StructType(Seq(StructField("same", IntegerType), StructField("same", LongType))))
    types.flatMap(nestedTypes).foreach { dt =>
      if (!dt.isInstanceOf[StructType]) {
        assert(supported(dt) == QueryPlanSerde.supportedDataType(dt), dt)
      }
    }
  }

  test("struct scalar subqueries retain supported fields and case-distinct names") {
    val distinct =
      StructType(Seq(StructField("a", IntegerType), StructField("A", LongType)))
    (scalarTypes.map(struct) ++ scalarTypes.map(dt => struct(struct(dt))) ++
      Seq(distinct, struct(distinct))).foreach { dt =>
      assert(supported(dt), dt)
      assert(CometScalarSubquery.convert(subquery(dt), Seq.empty, binding = false).isDefined, dt)
    }
  }

  test("struct scalar subqueries reject unsupported fields and shapes recursively") {
    val rejected = versionSpecificTypes ++ Seq(
      ArrayType(IntegerType),
      MapType(StringType, IntegerType),
      StructType(Nil),
      StructType(Seq(StructField("same", IntegerType), StructField("same", LongType))),
      CalendarIntervalType,
      YearMonthIntervalType(),
      DayTimeIntervalType(),
      CharType(5),
      VarcharType(5))
    rejected.foreach { dt =>
      Seq(struct(dt), struct(struct(dt))).foreach { nested =>
        assert(!supported(nested), nested)
      }
    }
    rejected.collect { case s: StructType => s }.foreach(s => assert(!supported(s), s))
  }

  test("struct decimal scale restrictions do not change the legacy non-struct gate") {
    withSQLConf("spark.sql.legacy.allowNegativeScaleOfDecimal" -> "true") {
      val negative = DecimalType(10, -2)
      assert(QueryPlanSerde.supportedDataType(struct(negative), allowComplex = true))
      assert(supported(negative))
      assert(!supported(struct(negative)))
      assert(!supported(struct(struct(negative))))
      Seq(DecimalType(1, 0), DecimalType(38, 0), DecimalType(38, 38)).foreach { dt =>
        assert(supported(struct(dt)), dt)
        assert(supported(struct(struct(dt))), dt)
      }
    }
  }

  test("nondeterministic variable-width consumers retain Spark fallback") {
    val condition = GreaterThanOrEqual(Rand(Literal(42L)), Literal(0.0))
    Seq(StringType, BinaryType, struct(StringType)).foreach { dt =>
      val field = GetStructField(subquery(struct(dt)), 0)
      val expr = If(condition, field, Literal.create(null, dt))
      assert(
        CometScalarSubquery.consumerFallbackReason(expr).exists(_.startsWith("Nondeterministic")))
      assert(QueryPlanSerde.exprToProtoInternal(expr, Seq.empty, binding = false).isEmpty)
    }
    val numeric = GetStructField(subquery(struct(LongType)), 0)
    assert(
      CometScalarSubquery.consumerFallbackReason(If(condition, numeric, Literal(0L))).isEmpty)
    val plain = If(condition, Literal("x"), Literal(""))
    assert(CometScalarSubquery.consumerFallbackReason(plain).isEmpty)
    val deterministic = OctetLength(GetStructField(subquery(struct(StringType)), 0))
    assert(CometScalarSubquery.consumerFallbackReason(deterministic).isEmpty)
  }

  test("synthetic NULL guards do not repeat stateful scalar consumers") {
    val condition = GreaterThan(Rand(Literal(42L)), Literal(0.5))
    val numeric = GetStructField(subquery(struct(LongType)), 0)
    val nullable = If(condition, numeric, Literal.create(null, LongType))
    val repeated = Coalesce(Seq(nullable, Literal(-1L)))
    assert(CometScalarSubquery.consumerFallbackReason(repeated).isDefined)
    assert(QueryPlanSerde.exprToProtoInternal(repeated, Seq.empty, binding = false).isEmpty)
    val ordinary =
      Coalesce(Seq(If(condition, Literal(1L), Literal.create(null, LongType)), Literal(-1L)))
    assert(CometScalarSubquery.consumerFallbackReason(ordinary).isEmpty)
    val plan = ProjectExec(Seq(Alias(numeric, "numeric")()), emptyInput)
    CometScalarSubquery.tagPlanInputs(plan)
    CometScalarSubquery.withPlanInputs(plan) {
      assert(CometScalarSubquery.consumerFallbackReason(ordinary).isDefined)
    }
    val wide = Alias(GetStructField(subquery(struct(StringType)), 0), "wide")()
    val producer = ProjectExec(Seq(wide), emptyInput)
    val consumer = ProjectExec(
      Seq(Alias(OctetLength(wide.toAttribute), "length")(), Alias(ordinary, "stateful")()),
      producer)
    CometScalarSubquery.tagPlanInputs(consumer)
    assert(CometScalarSubquery.rowPathFallbackReason(producer).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(consumer).isDefined)
    CometScalarSubquery.tagPlanInputs(producer)
    assert(CometScalarSubquery.rowPathFallbackReason(producer).isEmpty)
    assert(
      QueryPlanSerde
        .exprToProtoInternal(Alias(repeated, "result")(), Seq.empty, binding = false)
        .isEmpty)
    Seq(
      Coalesce(Seq(numeric, Literal(-1L))),
      Coalesce(Seq(numeric, If(condition, Literal(1L), Literal(2L))))).foreach { safe =>
      assert(CometScalarSubquery.consumerFallbackReason(safe).isEmpty, safe)
      assert(QueryPlanSerde.exprToProtoInternal(safe, Seq.empty, binding = false).isDefined, safe)
    }
  }

  test("row-dependent nested membership retains Spark short-circuit evaluation") {
    val dt = StructType(Seq(StructField("s", StringType), StructField("f", DoubleType)))
    val scalar = subquery(dt)
    val column = AttributeReference("candidate", dt)()
    val literal = Literal.create(Row("x", Double.NaN), dt)
    val foldable =
      CreateNamedStruct(Seq(Literal("s"), Literal("x"), Literal("f"), Literal(Double.NaN)))
    val dynamic = In(scalar, Seq(column))
    Seq(
      dynamic,
      In(column, Seq(scalar)),
      In(column, Seq(foldable, scalar)),
      If(Literal(true), dynamic, Literal(false)),
      Alias(dynamic, "membership")()).foreach { expr =>
      assert(
        CometScalarSubquery
          .consumerFallbackReason(expr)
          .exists(_.startsWith("Nested membership")),
        expr)
      assert(QueryPlanSerde.exprToProtoInternal(expr, Seq(column), binding = false).isEmpty)
    }
    Seq(In(scalar, Seq(literal)), In(scalar, Seq(subquery(dt))), In(column, Seq(literal)))
      .foreach { expr =>
        withSQLConf("spark.sql.subexpressionElimination.enabled" -> "false") {
          QueryPlanSerde.withSharedSubexpressions(Seq(expr)) {
            assert(CometScalarSubquery.consumerFallbackReason(expr).isEmpty, expr)
          }
        }
      }
    val source = Alias(scalar, "derived")()
    val child = ProjectExec(Seq(source), emptyInput)
    val derived = In(source.toAttribute, Seq(column))
    val parent = ProjectExec(Seq(Alias(derived, "result")()), child)
    CometScalarSubquery.tagPlanInputs(parent)
    CometScalarSubquery.withPlanInputs(parent) {
      assert(CometScalarSubquery.consumerFallbackReason(derived).isDefined)
      // The plan context also protects a sibling after a struct-producing conversion.
      assert(CometScalarSubquery.consumerFallbackReason(In(column, Seq(foldable))).isDefined)
      assert(
        CometScalarSubquery.consumerFallbackReason(In(source.toAttribute, Seq(literal))).isEmpty)
    }
    assert(CometScalarSubquery.consumerFallbackReason(In(column, Seq(foldable))).isEmpty)
    Seq(CreateArray(Seq(scalar)), CreateArray(Seq(CreateArray(Seq(scalar))))).foreach { nested =>
      val candidate = AttributeReference("nested", nested.dataType)()
      assert(CometScalarSubquery.consumerFallbackReason(In(nested, Seq(candidate))).isDefined)
      val derived = Alias(nested, "derived_nested")()
      val child = ProjectExec(Seq(derived), emptyInput)
      val expr = In(derived.toAttribute, Seq(candidate))
      val parent = ProjectExec(Seq(Alias(expr, "result")()), child)
      CometScalarSubquery.tagPlanInputs(parent)
      CometScalarSubquery.withPlanInputs(parent) {
        assert(CometScalarSubquery.consumerFallbackReason(expr).isDefined)
      }
    }
  }

  test("nullable scalar membership preserves skipped candidates with and without sharing") {
    val dt = StructType(Seq(StructField("s", StringType), StructField("f", DoubleType)))
    val scalar = subquery(dt)
    val literal = Literal.create(Row("x", 0.0d), dt)
    val bad = CreateNamedStruct(
      Seq(
        Literal("s"),
        Cast(Cast(GetStructField(subquery(struct(StringType)), 0), IntegerType), StringType),
        Literal("f"),
        Literal(0.0d)))
    val first = In(scalar, Seq(literal, bad))
    val second = In(scalar, Seq(scalar, bad))
    assert(first.references.isEmpty && second.references.isEmpty)
    assert(CometScalarSubquery.consumerFallbackReason(first).isDefined)
    Seq(false, true).foreach { shared =>
      withSQLConf("spark.sql.subexpressionElimination.enabled" -> shared.toString) {
        QueryPlanSerde.withSharedSubexpressions(Seq(first, second)) {
          Seq(first, second).foreach { expr =>
            assert(CometScalarSubquery.consumerFallbackReason(expr).isDefined)
            assert(QueryPlanSerde.exprToProtoInternal(expr, Seq.empty, binding = false).isEmpty)
          }
        }
      }
    }
  }

  test("comparison NULL short-circuit flags require shared-expression authorization") {
    val left = AttributeReference("left", StringType)()
    val bad = AttributeReference("bad", StringType)()
    val right = Cast(Cast(bad, IntegerType), StringType)
    val equal = EqualTo(left, right)
    val less = LessThan(left, right)
    val inputs = Seq(left, bad)
    def proto(expr: org.apache.spark.sql.catalyst.expressions.Expression): ExprOuterClass.Expr =
      QueryPlanSerde.exprToProtoInternal(expr, inputs, binding = false).get
    assert(!proto(equal).getEq.getNullShortCircuit)
    QueryPlanSerde.withSharedSubexpressions(Seq(equal)) {
      assert(proto(equal).getEq.getNullShortCircuit)
      assert(proto(Not(equal)).getNeq.getNullShortCircuit)
      assert(!proto(EqualNullSafe(left, right)).getEqNullSafe.getNullShortCircuit)
    }
    QueryPlanSerde.withSharedSubexpressions(Seq(equal, less), alwaysShared = true) {
      assert(!proto(equal).getEq.getNullShortCircuit)
      assert(!proto(less).getLt.getNullShortCircuit)
    }
  }

  test("opaque serialized consumers retain Spark fallback only for variable-width sources") {
    val text = GetStructField(subquery(struct(StringType)), 0)
    val numeric = GetStructField(subquery(struct(LongType)), 0)
    val opaque = ExprOuterClass.Expr
      .newBuilder()
      .setJvmScalarUdf(ExprOuterClass.JvmScalarUdf.newBuilder().setClassName("opaque"))
      .build()
    val parent = ExprOuterClass.Expr
      .newBuilder()
      .setScalarFunc(ExprOuterClass.ScalarFunc.newBuilder().setFunc("test").addArgs(opaque))
      .build()
    Seq(opaque, parent).foreach { proto =>
      assert(CometScalarSubquery.dispatchedConsumerFallbackReason(text, proto).isDefined)
      assert(CometScalarSubquery.dispatchedConsumerFallbackReason(numeric, proto).isEmpty)
      assert(CometScalarSubquery.dispatchedConsumerFallbackReason(Literal("x"), proto).isEmpty)
    }
    assert(
      CometScalarSubquery
        .dispatchedConsumerFallbackReason(text, ExprOuterClass.Expr.getDefaultInstance)
        .isEmpty)
  }

  test("calendar consumers require an allow-list even when their result is not a date") {
    val date = GetStructField(subquery(struct(DateType)), 0)
    val epoch = Literal.create(0, DateType)
    val difference = DateDiff(date, epoch)
    val unlisted = Seq(
      difference,
      DateAdd(date, Literal(1)),
      TruncDate(date, Literal("YEAR")),
      Cast(date, StringType),
      Coalesce(Seq(difference, Literal(0))),
      If(Literal(true), difference, Literal(0)))
    unlisted.foreach { expr =>
      assert(
        CometScalarSubquery
          .consumerFallbackReason(expr)
          .exists(_.contains("Calendar operations")),
        expr)
    }
    // Ordinary column/literal expressions must not acquire scalar-subquery provenance.
    assert(CometScalarSubquery.consumerFallbackReason(DateDiff(epoch, epoch)).isEmpty)
  }

  test("scalar-subquery guards record reasons in the current conversion scope") {
    val text = GetStructField(subquery(struct(StringType)), 0)
    val nondeterministic =
      If(GreaterThanOrEqual(Rand(Literal(42L)), Literal(0.0)), text, Literal(""))
    val calendar = TruncDate(GetStructField(subquery(struct(DateType)), 0), Literal("YEAR"))
    Seq(nondeterministic -> "Nondeterministic", calendar -> "Calendar operations")
      .foreach { case (expr, prefix) =>
        val (converted, reasons) =
          QueryPlanSerde.exprToProtoWithFallbackReasons(expr, Seq.empty, binding = false)
        assert(converted.isEmpty)
        assert(reasons.exists(_.startsWith(prefix)), reasons)
        assert(org.apache.comet.CometExplainInfo.currentFallbackReasons(expr).isEmpty)
      }
    val (converted, reasons) =
      QueryPlanSerde.exprToProtoWithFallbackReasons(Literal(42), Seq.empty, binding = false)
    assert(converted.nonEmpty)
    assert(reasons.isEmpty)
  }

  test("calendar allow-list retains raw transport comparison epoch and clock consumers") {
    val date = GetStructField(subquery(struct(DateType)), 0)
    val timestamp = GetStructField(subquery(struct(TimestampType)), 0)
    val ntz = GetStructField(subquery(struct(TimestampNTZType)), 0)
    val allowed = Seq(
      date,
      Alias(date, "renamed")(),
      Coalesce(Seq(date, Literal.create(0, DateType))),
      If(Literal(true), date, Literal.create(null, DateType)),
      EqualNullSafe(date, date),
      UnixDate(date),
      DayOfWeek(date),
      WeekDay(date),
      Cast(timestamp, LongType, Some("UTC")),
      Hour(timestamp, Some("+01:00")),
      Hour(ntz, Some("America/Los_Angeles")))
    allowed.foreach { expr =>
      assert(CometScalarSubquery.consumerFallbackReason(expr).isEmpty, expr)
    }
    assert(
      CometScalarSubquery
        .consumerFallbackReason(Hour(timestamp, Some("America/Los_Angeles")))
        .exists(_.contains("non-fixed timezones")))
  }

  private def clocks(child: Expression, zone: String): Seq[Expression] =
    Seq(Hour(child, Some(zone)), Minute(child, Some(zone)), Second(child, Some(zone)))

  private def nativeClockInput(
      expr: Expression,
      proto: ExprOuterClass.Expr): ExprOuterClass.Expr =
    expr match {
      case _: Hour =>
        assert(proto.hasHour)
        proto.getHour.getChild
      case _: Minute =>
        assert(proto.hasMinute)
        proto.getMinute.getChild
      case _: Second =>
        assert(proto.hasSecond)
        proto.getSecond.getChild
      case other => fail(s"Expected a clock expression, got $other")
    }

  test("clock offsets with seconds dispatch or retain a clear Spark fallback") {
    val timestamp = AttributeReference("ts", TimestampType)()
    val zone = "+05:45:30"
    for (enabled <- Seq(false, true)) {
      withSQLConf(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key -> enabled.toString) {
        clocks(timestamp, zone).foreach { clock =>
          val (proto, reasons) =
            QueryPlanSerde.exprToProtoWithFallbackReasons(Alias(clock, "clock")(), Seq(timestamp))
          if (enabled) {
            assert(proto.exists(_.hasJvmScalarUdf), s"$clock: $proto $reasons")
            assert(reasons.isEmpty, reasons)
          } else {
            assert(proto.isEmpty, clock)
            assert(reasons.contains(CometTimeZone.unsupportedReason(Some(zone))), reasons)
            assert(
              reasons.exists(_.contains(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key)),
              reasons)
          }
        }
      }
    }
  }

  test("ordinary computed regional clocks dispatch before native serialization") {
    val id = AttributeReference("id", IntegerType)()
    val timestamp = AttributeReference("ts", TimestampType)()
    val scalar = Literal.create(9089364630896000000L, TimestampType)
    val seconds = AttributeReference("seconds", LongType)()
    val date = AttributeReference("date", DateType)()
    val scalarNull = Cast(date, LongType, Some("UTC"), EvalMode.LEGACY)
    val predicate = GreaterThan(id, Literal(0))
    val conditional = If(predicate, scalar, timestamp)
    val conditionalSeconds = If(predicate, Literal(1800L), seconds)
    for (enabled <- Seq(false, true)) {
      withSQLConf(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key -> enabled.toString) {
        val children = Seq(
          scalar,
          Literal.create(null, TimestampType),
          conditional,
          CaseWhen(Seq(predicate -> scalar), Some(timestamp)),
          Coalesce(Seq(timestamp, scalar)),
          Cast(conditional, TimestampType, Some("UTC")),
          Cast(scalarNull, TimestampType, Some("UTC")),
          SecondsToTimestamp(conditionalSeconds),
          PreciseTimestampConversion(conditionalSeconds, LongType, TimestampType),
          TruncTimestamp(Literal("HOUR"), conditional, Some("America/Los_Angeles")),
          FromUTCTimestamp(conditional, Literal("UTC")),
          ToUTCTimestamp(conditional, Literal("UTC")))
        children.flatMap(clocks(_, "America/Los_Angeles")).foreach { clock =>
          // There is no subquery: the general serde gate must reject every scalar-capable shape.
          assert(CometScalarSubquery.consumerFallbackReason(clock).isEmpty)
          val (proto, reasons) = QueryPlanSerde
            .exprToProtoWithFallbackReasons(
              Alias(clock, "clock")(),
              Seq(id, timestamp, seconds, date))
          if (enabled) {
            assert(proto.exists(_.hasJvmScalarUdf), s"$clock: $proto")
            assert(reasons.isEmpty, reasons)
          } else {
            assert(proto.isEmpty, clock)
            assert(reasons.exists(_.contains(CometTimeZone.clockInputUnsupportedReason)), reasons)
            assert(
              reasons.exists(_.contains(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key)),
              reasons)
          }
        }
      }
    }
  }

  test("clock serialization retains column arrays and full-range fixed-offset and NTZ inputs") {
    val id = AttributeReference("id", IntegerType)()
    for {
      enabled <- Seq(false, true)
      dt <- Seq(TimestampType, TimestampNTZType)
      zone <- Seq("UTC", "+01:00", "America/Los_Angeles")
    } {
      withSQLConf(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key -> enabled.toString) {
        val timestamp = AttributeReference("ts", dt)()
        val scalar = Literal.create(Long.MaxValue, dt)
        val computed = Seq(
          scalar,
          Literal.create(Long.MinValue, dt),
          Literal.create(null, dt),
          If(GreaterThan(id, Literal(0)), scalar, timestamp))
        val children =
          if (dt == TimestampNTZType || zone != "America/Los_Angeles") timestamp +: computed
          else Seq(timestamp)
        children.foreach { child =>
          clocks(child, zone).foreach { clock =>
            val (proto, reasons) = QueryPlanSerde.exprToProtoWithFallbackReasons(
              Alias(clock, "clock")(),
              Seq(id, timestamp))
            assert(proto.isDefined, s"$clock: $reasons")
            assert(reasons.isEmpty, reasons)
            val input = nativeClockInput(clock, proto.get)
            if (child eq timestamp) assert(input.hasBound, input)
          }
        }
        // Native input binding is emitted by AttributeReference's serde. A raw BoundReference
        // has no standalone serde, but must remain accepted by the clock support-level gate.
        val bound = BoundReference(0, dt, nullable = true)
        assert(CometHour.getSupportLevel(Hour(bound, Some(zone))) == Compatible())
        assert(CometMinute.getSupportLevel(Minute(bound, Some(zone))) == Compatible())
        assert(CometSecond.getSupportLevel(Second(bound, Some(zone))) == Compatible())
      }
    }
  }

  test("regional clocks retain array shapes through timestamp producers") {
    for {
      enabled <- Seq(false, true)
      allowIncompatible <- Seq(false, true)
      zone <- Seq("America/Los_Angeles", "Asia/Kolkata")
    } {
      val optIns = Seq("TruncTimestamp", "FromUTCTimestamp", "ToUTCTimestamp")
        .map { name =>
          s"spark.comet.expression.$name.allowIncompatible" -> allowIncompatible.toString
        }
      val confs = Seq(
        CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key -> enabled.toString,
        "spark.sql.ansi.enabled" -> "false") ++ optIns
      withSQLConf(confs: _*) {
        val timestamp = AttributeReference("ts", TimestampType)()
        val inputs = Seq(
          StringType,
          DateType,
          TimestampNTZType,
          IntegerType,
          LongType,
          FloatType,
          DoubleType,
          DecimalType(20, 6),
          BooleanType).map(dt => AttributeReference(s"input_${dt.typeName}", dt)())
        val nativeSeconds = inputs.filter(a =>
          Seq(IntegerType, LongType, FloatType, DoubleType).contains(a.dataType))
        val casts = inputs.map(Cast(_, TimestampType, Some(zone)))
        val micros = inputs.find(_.dataType == LongType).get
        val precise = PreciseTimestampConversion(micros, LongType, TimestampType)
        val decimal = inputs.find(_.dataType.isInstanceOf[DecimalType]).get
        val arrayProducers =
          (nativeSeconds.map(SecondsToTimestamp(_)) ++ casts ++ Seq(
            precise,
            Cast(SecondsToTimestamp(micros), TimestampType, Some(zone))))
            .map(_ -> false)
        val text = inputs.find(_.dataType == StringType).get
        val scalar = Literal.create(0L, TimestampType)
        val optInProducers = Seq(
          TruncTimestamp(Literal("HOUR"), timestamp, Some(zone)),
          TruncTimestamp(Literal.create(null, StringType), timestamp, Some(zone)),
          TruncTimestamp(text, scalar, Some(zone)),
          FromUTCTimestamp(timestamp, Literal("Asia/Tokyo")),
          FromUTCTimestamp(scalar, text),
          ToUTCTimestamp(timestamp, Literal("Asia/Tokyo")),
          ToUTCTimestamp(scalar, text)).map(_ -> !allowIncompatible)
        val producers = arrayProducers ++ optInProducers ++
          Seq(SecondsToTimestamp(decimal) -> true)
        producers.foreach { case (producer, dispatchedChild) =>
          clocks(producer, zone).foreach { clock =>
            val (proto, reasons) = QueryPlanSerde.exprToProtoWithFallbackReasons(
              Alias(clock, "clock")(),
              timestamp +: inputs)
            if (dispatchedChild && !enabled) {
              assert(proto.isEmpty, clock)
              assert(
                reasons.exists(_.contains(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key)),
                reasons)
              assert(
                !reasons.exists(_.contains(CometTimeZone.clockInputUnsupportedReason)),
                reasons)
            } else {
              assert(proto.isDefined, s"$clock: $reasons")
              assert(reasons.isEmpty, reasons)
              val input = nativeClockInput(clock, proto.get)
              assert(input.hasJvmScalarUdf == dispatchedChild, s"$clock: $input")
            }
          }
        }
      }
    }
  }

  test("unix_timestamp keeps date columns native and dispatches potentially scalar dates") {
    val id = AttributeReference("id", IntegerType)()
    for {
      enabled <- Seq(false, true)
      zone <- Seq("UTC", "+01:00", "America/Los_Angeles")
      dt <- Seq(DateType, TimestampType, TimestampNTZType)
    } {
      withSQLConf(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key -> enabled.toString) {
        val column = AttributeReference("value", dt)()
        val scalar =
          if (dt == DateType) Literal.create(Int.MaxValue, dt)
          else Literal.create(Long.MaxValue, dt)
        val children = Seq(
          column,
          scalar,
          Literal.create(null, dt),
          If(GreaterThan(id, Literal(0)), scalar, column))
        children.foreach { child =>
          val unix = UnixTimestamp(child, Literal("yyyy-MM-dd"), Some(zone), failOnError = false)
          assert(CometScalarSubquery.consumerFallbackReason(unix).isEmpty)
          val (proto, reasons) =
            QueryPlanSerde.exprToProtoWithFallbackReasons(Alias(unix, "epoch")(), Seq(id, column))
          if (dt != DateType || (child eq column)) {
            assert(proto.exists(_.hasUnixTimestamp), s"$unix: $proto $reasons")
            assert(reasons.isEmpty, reasons)
            if (child eq column) assert(proto.get.getUnixTimestamp.getChild.hasBound)
          } else if (enabled) {
            assert(proto.exists(_.hasJvmScalarUdf), s"$unix: $proto $reasons")
            assert(reasons.isEmpty, reasons)
          } else {
            assert(proto.isEmpty, unix)
            assert(
              reasons.exists(_.contains(
                "date inputs that may be scalar require Spark for checked timezone conversion")),
              reasons)
            assert(
              reasons.exists(_.contains(CometConf.COMET_SCALA_UDF_CODEGEN_ENABLED.key)),
              reasons)
          }
        }
      }
    }
  }

  test("wide scalar aliases retain a Spark row path through consumers") {
    val text = GetStructField(subquery(struct(StringType)), 0)
    val alias = Alias(text, "wide")()
    val producer = ProjectExec(Seq(alias), emptyInput)
    val consumer = FilterExec(IsNotNull(alias.toAttribute), producer)
    val reduced = Alias(OctetLength(alias.toAttribute), "length")()
    val reducer = ProjectExec(Seq(reduced), consumer)
    val ancestor = ProjectExec(Seq(Alias(reduced.toAttribute, "result")()), reducer)
    CometScalarSubquery.tagPlanInputs(ancestor)
    assert(CometScalarSubquery.rowPathFallbackReason(producer).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(consumer).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(reducer).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(ancestor).isEmpty)
    // Recompute on the same node: a standalone scalar projection retains its native contract.
    CometScalarSubquery.tagPlanInputs(producer)
    assert(CometScalarSubquery.rowPathFallbackReason(producer).isEmpty)
  }

  test("opaque siblings protect both consumers and producers of retained wide values") {
    val text = Alias(GetStructField(subquery(struct(StringType)), 0), "text")()
    val binary = AttributeReference("binary", BinaryType)()
    Seq[Expression](Rand(Literal(42L)), OctetLength(binary)).foreach { sibling =>
      val producer = ProjectExec(Seq(text), emptyInput)
      val consumer = ProjectExec(
        Seq(Alias(OctetLength(text.toAttribute), "length")(), Alias(sibling, "sibling")()),
        producer)
      CometScalarSubquery.tagPlanInputs(consumer)
      assert(CometScalarSubquery.rowPathFallbackReason(producer).isDefined, sibling)
      assert(CometScalarSubquery.rowPathFallbackReason(consumer).isDefined, sibling)
      val inner = ProjectExec(Seq(text, Alias(sibling, "sibling")()), emptyInput)
      val outer =
        ProjectExec(Seq(Alias(OctetLength(text.toAttribute), "length")(), inner.output(1)), inner)
      CometScalarSubquery.tagPlanInputs(outer)
      assert(CometScalarSubquery.rowPathFallbackReason(inner).isDefined, sibling)
      assert(CometScalarSubquery.rowPathFallbackReason(outer).isDefined, sibling)
    }
  }

  test("pure projections preserve bounded native producer batches") {
    val scalar = Alias(subquery(struct(StringType)), "scalar")()
    val producer = ProjectExec(Seq(scalar), emptyInput)
    val field = Alias(GetStructField(scalar.toAttribute, 0), "text")()
    val fields = ProjectExec(Seq(field), producer)
    val reducer = ProjectExec(Seq(Alias(OctetLength(field.toAttribute), "length")()), fields)
    CometScalarSubquery.tagPlanInputs(reducer)
    Seq(producer, fields, reducer).foreach { plan =>
      assert(CometScalarSubquery.rowPathFallbackReason(plan).isEmpty, plan)
    }
  }

  test("a retained scalar alias cannot grow an unbounded projection intermediate") {
    Seq[DataType](StringType, LongType).foreach { dt =>
      val scalar = Alias(GetStructField(subquery(struct(dt)), 0), "scalar")()
      val producer = ProjectExec(Seq(scalar), emptyInput)
      val repeated = if (dt == StringType) {
        StringRepeat(scalar.toAttribute, Literal(2048))
      } else {
        StringRepeat(Literal("x"), Cast(scalar.toAttribute, IntegerType))
      }
      val reducer = ProjectExec(Seq(Alias(OctetLength(repeated), "length")()), producer)
      CometScalarSubquery.tagPlanInputs(reducer)
      assert(
        CometScalarSubquery.rowPathFallbackReason(producer).isDefined == (dt == StringType),
        dt)
      assert(CometScalarSubquery.rowPathFallbackReason(reducer).isDefined, dt)
    }
  }

  test("retained alias roots check growing constant branches and bounded choices") {
    val number = Alias(GetStructField(subquery(struct(LongType)), 0), "number")()
    val id = emptyInput.output.head
    val producer = ProjectExec(Seq(number, id), emptyInput)
    val growing = If(
      IsNotNull(id),
      StringRepeat(Literal("x"), Literal(512000)),
      Cast(number.toAttribute, StringType))
    val reducer = ProjectExec(Seq(Alias(OctetLength(growing), "length")()), producer)
    CometScalarSubquery.tagPlanInputs(reducer)
    assert(CometScalarSubquery.rowPathFallbackReason(producer).isEmpty)
    assert(CometScalarSubquery.rowPathFallbackReason(reducer).isDefined)

    val predicate = FilterExec(IsNotNull(OctetLength(growing)), producer)
    CometScalarSubquery.tagPlanInputs(predicate)
    assert(CometScalarSubquery.rowPathFallbackReason(predicate).isDefined)
    val topK = org.apache.spark.sql.execution.TakeOrderedAndProjectExec(
      8192,
      Seq.empty,
      Seq(Alias(OctetLength(growing), "length")()),
      producer)
    CometScalarSubquery.tagPlanInputs(topK)
    assert(CometScalarSubquery.rowPathFallbackReason(topK).isDefined)

    val bounded = If(IsNotNull(id), Cast(id, StringType), Cast(number.toAttribute, StringType))
    val native = ProjectExec(Seq(Alias(bounded, "text")()), producer)
    val consumer = FilterExec(IsNotNull(native.output.head), native)
    CometScalarSubquery.tagPlanInputs(consumer)
    Seq(producer, native, consumer).foreach { plan =>
      assert(CometScalarSubquery.rowPathFallbackReason(plan).isEmpty, plan)
    }
  }

  test("retained numeric aliases check window values without evaluating frame data types") {
    import org.apache.spark.sql.catalyst.expressions.{CurrentRow, RowFrame, RowNumber, SpecifiedWindowFrame, UnboundedPreceding, WindowExpression, WindowSpecDefinition}
    import org.apache.spark.sql.catalyst.expressions.aggregate.CollectList
    import org.apache.spark.sql.execution.window.WindowExec
    val number = Alias(GetStructField(subquery(struct(LongType)), 0), "number")()
    val producer = ProjectExec(Seq(number), emptyInput)
    val spec = WindowSpecDefinition(
      Seq(number.toAttribute),
      Seq.empty,
      SpecifiedWindowFrame(RowFrame, UnboundedPreceding, CurrentRow))
    val rank = WindowExec(
      Seq(Alias(WindowExpression(RowNumber(), spec), "rank")()),
      Seq(number.toAttribute),
      Seq.empty,
      producer)
    CometScalarSubquery.tagPlanInputs(rank)
    assert(CometScalarSubquery.rowPathFallbackReason(rank).isEmpty)
    val collect = WindowExec(
      Seq(
        Alias(
          WindowExpression(CollectList(number.toAttribute).toAggregateExpression(), spec),
          "values")()),
      Seq(number.toAttribute),
      Seq.empty,
      producer)
    CometScalarSubquery.tagPlanInputs(collect)
    assert(CometScalarSubquery.rowPathFallbackReason(collect).isDefined)
  }

  test("float field formatting preserves the native struct projection contract") {
    val dt = StructType(Seq(StructField("text", StringType), StructField("number", DoubleType)))
    val scalar = Alias(subquery(dt), "scalar")()
    val producer = ProjectExec(Seq(scalar), emptyInput)
    val formatted = Cast(GetStructField(scalar.toAttribute, 1), StringType)
    val consumer = ProjectExec(Seq(Alias(formatted, "formatted")()), producer)
    CometScalarSubquery.tagPlanInputs(consumer)
    assert(CometScalarSubquery.rowPathFallbackReason(producer).isEmpty)
    assert(CometScalarSubquery.rowPathFallbackReason(consumer).isEmpty)
  }

  test("bounded scalar aliases do not force a Spark row path") {
    val number = GetStructField(subquery(struct(LongType)), 0)
    val values =
      Seq(number, Cast(number, StringType), CreateArray(Seq(number, Literal(1L), Literal(2L))))
    values.foreach { value =>
      val alias = Alias(value, "bounded")()
      val producer = ProjectExec(Seq(alias), emptyInput)
      val consumer = FilterExec(IsNotNull(alias.toAttribute), producer)
      CometScalarSubquery.tagPlanInputs(consumer)
      assert(CometScalarSubquery.rowPathFallbackReason(producer).isEmpty, value)
      assert(CometScalarSubquery.rowPathFallbackReason(consumer).isEmpty, value)
      CometScalarSubquery.withPlanInputs(consumer) {
        assert(
          CometScalarSubquery.operatorArgumentFallbackReason(alias.toAttribute).isEmpty,
          value)
      }
    }
  }

  test("wide row paths mark only the affected union branch") {
    import org.apache.spark.sql.execution.UnionExec
    val text = Alias(GetStructField(subquery(struct(StringType)), 0), "value")()
    val wide = ProjectExec(Seq(text), emptyInput)
    val number = GetStructField(subquery(struct(LongType)), 0)
    val boundedText = Alias(Cast(number, StringType), "value")()
    val bounded = ProjectExec(Seq(boundedText), emptyInput)
    val union = UnionExec(Seq(wide, bounded))
    val consumer = ProjectExec(Seq(Alias(OctetLength(union.output.head), "length")()), union)
    CometScalarSubquery.tagPlanInputs(consumer)
    assert(CometScalarSubquery.rowPathFallbackReason(wide).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(union).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(consumer).isDefined)
    assert(CometScalarSubquery.rowPathFallbackReason(bounded).isEmpty)
  }

  test("reused exchanges remap source bounds without depending on attribute identity") {
    import org.apache.spark.sql.catalyst.plans.physical.SinglePartition
    import org.apache.spark.sql.execution.exchange.{ReusedExchangeExec, ShuffleExchangeExec}
    val number = GetStructField(subquery(struct(LongType)), 0)
    Seq(Cast(number, StringType), GetStructField(subquery(struct(StringType)), 0))
      .foreach { value =>
        val alias = Alias(value, "value")()
        val producer = ProjectExec(Seq(alias), emptyInput)
        val exchange = ShuffleExchangeExec(SinglePartition, producer)
        val reused = ReusedExchangeExec(exchange.output.map(_.newInstance()), exchange)
        val consumer = FilterExec(IsNotNull(reused.output.head), reused)
        CometScalarSubquery.tagPlanInputs(consumer)
        val wide = value.dataType == StringType && !value.isInstanceOf[Cast]
        assert(CometScalarSubquery.rowPathFallbackReason(producer).isDefined == wide, value)
        assert(CometScalarSubquery.rowPathFallbackReason(consumer).isDefined == wide, value)
        CometScalarSubquery.withPlanInputs(consumer) {
          assert(
            CometScalarSubquery
              .operatorArgumentFallbackReason(reused.output.head)
              .isDefined == wide,
            value)
        }
      }
  }

  test("shared Bloom keys retain lazy NO_CODEGEN evaluation despite non-null-intolerant class") {
    val input = AttributeReference("s", StringType)()
    val key = Cast(input, LongType, None, EvalMode.ANSI)
    val filter = GetStructField(subquery(struct(BinaryType)), 0)
    val first = BloomFilterMightContain(filter, key)
    val second = BloomFilterMightContain(Literal.create(null, NullType), key)
    for (mode <- Seq("CODEGEN_ONLY", "NO_CODEGEN"); enabled <- Seq(false, true)) {
      withSQLConf(
        "spark.sql.codegen.factoryMode" -> mode,
        "spark.sql.subexpressionElimination.enabled" -> enabled.toString) {
        QueryPlanSerde.withSharedSubexpressions(Seq(first, second), eagerSharing = true) {
          val proto = QueryPlanSerde.exprToProtoInternal(first, Seq(input), binding = false)
          assert(proto.isEmpty)
          val lower = CometBloomFilterMightContain.convert(first, Seq(input), binding = false)
          assert(lower.isDefined)
          assert(lower.get.getBloomFilterMightContain.getNullShortCircuit == !enabled)
          assert(
            QueryPlanSerde.isEagerSharedSubexpression(key) == (enabled && mode == "CODEGEN_ONLY"))
        }
      }
    }
  }

  test("padding permits only whole eager argument roots and retains NO_CODEGEN masks") {
    val id = AttributeReference("id", LongType)()
    val text = AttributeReference("text", StringType)()
    val condition = AttributeReference("p", BooleanType)()
    val field = GetStructField(subquery(struct(StringType)), 0)
    val string = If(condition, field, text)
    val inner = Add(id, Literal(3L))
    val length = Cast(inner, IntegerType)
    val left = StringLPad(string, length, Literal("x"))
    val right = StringRPad(string, length, Literal("x"))
    val inputs = Seq(id, text, condition)
    for (mode <- Seq("CODEGEN_ONLY", "NO_CODEGEN"); enabled <- Seq(false, true)) {
      withSQLConf(
        "spark.sql.codegen.factoryMode" -> mode,
        "spark.sql.subexpressionElimination.enabled" -> enabled.toString) {
        QueryPlanSerde.withSharedSubexpressions(Seq(left, right), eagerSharing = true) {
          Seq(left, right).foreach { padding =>
            // The enclosing struct guard rejects an unknown length. Check the lower serializer's
            // independent NULL-mask contract without using that fallback as its proof.
            assert(QueryPlanSerde.exprToProtoInternal(padding, inputs, binding = false).isEmpty)
            val proto = padding match {
              case p: StringLPad => CometStringLPad.convert(p, inputs, binding = false)
              case p: StringRPad => CometStringRPad.convert(p, inputs, binding = false)
            }
            assert(proto.isDefined)
            assert(proto.get.getScalarFunc.getNullShortCircuit == !enabled)
            assert(
              QueryPlanSerde.isEagerSharedSubexpression(
                length) == (enabled && mode == "CODEGEN_ONLY"))
          }
        }
        QueryPlanSerde.withSharedSubexpressions(Seq(left, inner), eagerSharing = true) {
          val proto = CometStringLPad.convert(left, inputs, binding = false)
          assert(proto.isDefined)
          assert(proto.get.getScalarFunc.getNullShortCircuit == !enabled)
          assert(!QueryPlanSerde.isEagerSharedSubexpression(length))
        }
      }
    }
  }

}
