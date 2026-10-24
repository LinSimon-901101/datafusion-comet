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

package org.apache.spark.sql.benchmark

import org.apache.spark.benchmark.Benchmark
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.{GetStructField, Literal, ScalarSubquery}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType

import org.apache.comet.CometConf

/**
 * Measures bounded scalar-subquery consumers and an ordinary typed NULL struct control. To run:
 * {{{
 *   SPARK_GENERATE_BENCHMARK_FILES=1 make benchmark-org.apache.spark.sql.benchmark.CometScalarSubqueryConsumerBenchmark
 * }}}
 * An optional first argument sets the row count (default 1M). Each result is a consumed checksum.
 */
object CometScalarSubqueryConsumerBenchmark extends CometBenchmarkBase {

  private val cases = Seq(
    (
      "ordinary NULL struct -> array -> hash",
      "SELECT sum(CAST(hash(array(IF(p, payload, " +
        "CAST(NULL AS STRUCT<k:BIGINT>)))) AS BIGINT)) FROM consumer_outer",
      false),
    (
      "merged subqueries -> integer cast -> length",
      "SELECT sum(CAST(octet_length(CAST(payload.k + " +
        "(SELECT max(n) FROM consumer_cast_source) AS STRING)) AS BIGINT)), " +
        "sum((SELECT min(n) FROM consumer_cast_source)) FROM consumer_outer",
      true),
    (
      "merged subqueries -> three BIGINT array -> hash",
      "SELECT sum(CAST(hash(array(payload.k, " +
        "CAST((SELECT max(id) FROM consumer_array_source) AS BIGINT), " +
        "CAST((SELECT min(id) FROM consumer_array_source) AS BIGINT))) AS BIGINT)) " +
        "FROM consumer_outer",
      true))

  private def configs(comet: Boolean): Seq[(String, String)] = Seq(
    CometConf.COMET_ENABLED.key -> comet.toString,
    CometConf.COMET_EXEC_ENABLED.key -> comet.toString)

  private def consume(query: String, comet: Boolean): Seq[Row] =
    withSQLConf(configs(comet): _*) {
      spark.sql(query).collect().toSeq
    }

  private def measure(name: String, query: String, merged: Boolean, rows: Long): Unit = {
    runBenchmark(name) {
      val benchmark = new Benchmark(name, rows, output = output)
      val expected = consume(query, comet = false)
      withSQLConf(configs(comet = true): _*) {
        val df = spark.sql(query)
        val actual = df.collect().toSeq
        require(actual == expected, s"$name: Comet $actual, Spark $expected")
        val hasMergedStruct = df.queryExecution.optimizedPlan.exists { plan =>
          plan.expressions.exists(_.exists {
            case GetStructField(s: ScalarSubquery, _, _) =>
              s.dataType.isInstanceOf[StructType]
            case _ => false
          })
        }
        require(hasMergedStruct == merged, df.queryExecution.optimizedPlan.toString)
        if (!merged) {
          require(
            df.queryExecution.optimizedPlan.exists(_.expressions.exists(_.exists {
              case Literal(null, _: StructType) => true
              case _ => false
            })),
            "Expected an ordinary typed NULL struct")
        }
        val plan = stripAQEPlan(df.queryExecution.executedPlan)
        require(findFirstNonCometOperator(plan).isEmpty, s"Expected native $name:\n$plan")
        benchmark.out.println(s"Verified checksum: $actual")
        benchmark.out.println(s"Spark: ${spark.version}, rows: $rows, batch size: 8192")
        benchmark.out.println(s"Native library path: ${System.getProperty("java.library.path")}")
        benchmark.out.println(plan.treeString)
      }
      // Keep normal constant folding: it creates the ordinary typed NULL struct control.
      Seq("Spark" -> false, "Comet" -> true, "Comet (repeat)" -> true).foreach {
        case (label, comet) => benchmark.addCase(label)(_ => consume(query, comet))
      }
      benchmark.run()
    }
  }

  override def runCometBenchmark(mainArgs: Array[String]): Unit = {
    val rows = mainArgs.headOption.map(_.toLong).getOrElse(1024L * 1024)
    require(rows > 0, "row count must be positive")
    withSQLConf(
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
      SQLConf.SUBQUERY_REUSE_ENABLED.key -> "true",
      SQLConf.SHUFFLE_PARTITIONS.key -> "1",
      SQLConf.ANSI_ENABLED.key -> "true",
      CometConf.COMET_BATCH_SIZE.key -> "8192") {
      withTempPath { dir =>
        withTempTable("consumer_outer", "consumer_cast_source", "consumer_array_source") {
          withSQLConf(configs(comet = false): _*) {
            spark
              .range(0, rows, 1, 1)
              .selectExpr("id % 2 = 0 AS p", "named_struct('k', id) AS payload")
              .write
              .parquet(s"${dir.getCanonicalPath}/outer")
            spark
              .range(1)
              .selectExpr("CAST(262144 AS BIGINT) AS n")
              .coalesce(1)
              .write
              .parquet(s"${dir.getCanonicalPath}/cast_source")
            spark.range(0, 2, 1, 1).write.parquet(s"${dir.getCanonicalPath}/array_source")
          }
          Seq(
            "consumer_outer" -> "outer",
            "consumer_cast_source" -> "cast_source",
            "consumer_array_source" -> "array_source").foreach { case (table, path) =>
            spark.read.parquet(s"${dir.getCanonicalPath}/$path").createOrReplaceTempView(table)
          }
          cases.foreach { case (name, query, merged) => measure(name, query, merged, rows) }
        }
      }
    }
  }
}
