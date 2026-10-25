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

package org.apache.comet.exec

import scala.jdk.CollectionConverters._

import org.apache.spark.SparkConf
import org.apache.spark.sql.{CometTestBase, Row}
import org.apache.spark.sql.comet.CometTakeOrderedAndProjectExec
import org.apache.spark.sql.internal.SQLConf

import org.apache.comet.CometConf
import org.apache.comet.CometSparkSessionExtensions.isSpark40Plus
import org.apache.comet.serde.OperatorOuterClass.Operator

class CometTopKMemorySuite extends CometTestBase {
  override protected def sparkConf: SparkConf = super.sparkConf
    .set("spark.memory.offHeap.size", "6m")
    .set(CometConf.COMET_ONHEAP_ENABLED.key, "false")
    .set(CometConf.COMET_OFFHEAP_MEMORY_POOL_TYPE.key, "fair_unified")
    .set(CometConf.COMET_OFFHEAP_MEMORY_POOL_FRACTION.key, "0.5")
    .set(CometConf.COMET_BATCH_SIZE.key, "8192")

  private def operators(op: Operator): Seq[Operator] =
    Seq(op) ++ op.getChildrenList.asScala.toSeq.flatMap(operators)

  if (isSpark40Plus) {
    for (wide <- Seq(false, true)) {
      test(s"deferred native TopK bounds key and payload workspace: wide=$wide") {
        withSQLConf(
          // The effective native budgets reproduce the 512KiB and 3MiB failures.
          CometConf.COMET_OFFHEAP_MEMORY_POOL_FRACTION.key ->
            (if (wide) "0.5" else (1.0 / 12).toString),
          SQLConf.ANSI_ENABLED.key -> "true",
          SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false",
          SQLConf.LEAF_NODE_DEFAULT_PARALLELISM.key -> "1") {
          withTempPath { path =>
            withSQLConf(CometConf.COMET_ENABLED.key -> "false") {
              spark
                .range(16384)
                .coalesce(1)
                .selectExpr(
                  "CAST(id AS INT) AS k",
                  if (wide) "rpad(CAST(id AS STRING), 256, '0') AS payload"
                  else "CAST(16383 - id AS INT) AS payload")
                .write
                .parquet(path.getCanonicalPath)
            }
            withParquetTable(path.getCanonicalPath, "topk_memory") {
              withParquetTable(Seq(Tuple1(1)), "topk_one") {
                val queries = Seq(
                  "k + (SELECT max(_1) FROM topk_one)",
                  "k + (SELECT max(named_struct('v', _1)) FROM topk_one).v",
                  "k + (SELECT max(_1) FROM topk_one), " +
                    "k + (SELECT min(_1) FROM topk_one)")
                val expected = (0 until 7).map { i =>
                  if (wide) Row(i.toString.padTo(256, '0')) else Row(16383 - i)
                }
                queries.foreach { keys =>
                  val query = s"SELECT payload FROM topk_memory ORDER BY $keys LIMIT 7"
                  val (_, plan) =
                    checkSparkAnswerAndOperator(
                      sql(query),
                      classOf[CometTakeOrderedAndProjectExec])
                  val topK = collect(plan) { case t: CometTakeOrderedAndProjectExec => t }.head
                  assert(operators(topK.finalNativePlan(1).get).exists { op =>
                    op.hasSort && op.getSort.getDeferSortErrors
                  })
                  val df = sql(query)
                  val actual = df.queryExecution.toRdd
                    .collect()
                    .map { row =>
                      if (wide) Row(row.getUTF8String(0).toString) else Row(row.getInt(0))
                    }
                    .toSeq
                  assert(actual == expected)
                  assert(collect(df.queryExecution.executedPlan) {
                    case t: CometTakeOrderedAndProjectExec => t
                  }.nonEmpty)
                }
              }
            }
          }
        }
      }
    }
  }
}
