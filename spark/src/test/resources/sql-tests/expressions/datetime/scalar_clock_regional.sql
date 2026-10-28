-- Licensed to the Apache Software Foundation (ASF) under one
-- or more contributor license agreements.  See the NOTICE file
-- distributed with this work for additional information
-- regarding copyright ownership.  The ASF licenses this file
-- to you under the Apache License, Version 2.0 (the
-- "License"); you may not use this file except in compliance
-- with the License.  You may obtain a copy of the License at
--
--   http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing,
-- software distributed under the License is distributed on an
-- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
-- KIND, either express or implied.  See the License for the
-- specific language governing permissions and limitations
-- under the License.

-- Config: spark.sql.session.timeZone=America/Los_Angeles
-- Config: spark.sql.adaptive.enabled=false
-- Config: spark.sql.execution.reuseSubquery=false
-- Config: spark.comet.exec.scalaUDF.codegen.enabled=false

statement
CREATE TABLE scalar_clock_regional(id INT, ts TIMESTAMP, ntz TIMESTAMP_NTZ) USING parquet

statement
INSERT INTO scalar_clock_regional VALUES
  (1, TIMESTAMP '2024-07-01 12:34:56', TIMESTAMP_NTZ '2024-07-01 12:34:56'),
  (2, TIMESTAMP '2024-01-01 12:34:56', NULL)

-- Ordinary regional columns retain the existing native kernel.
query expect_native(hour,minute,second)
SELECT hour(ts), minute(ts), second(ts) FROM scalar_clock_regional

query expect_native(hour,minute,second,unix_timestamp)
SELECT id, hour((SELECT max(ntz) FROM scalar_clock_regional)),
       minute((SELECT max(ntz) FROM scalar_clock_regional)),
       second((SELECT max(ntz) FROM scalar_clock_regional)),
       unix_timestamp((SELECT max(ts) FROM scalar_clock_regional))
FROM scalar_clock_regional

query expect_fallback(Clock expressions with inputs that may be scalar require Spark for non-fixed timezones)
SELECT id, hour((SELECT max(ts) FROM scalar_clock_regional)),
       minute((SELECT max(ts) FROM scalar_clock_regional)),
       second((SELECT max(ts) FROM scalar_clock_regional))
FROM scalar_clock_regional

-- Year 290000 is outside chrono's range. Sampling an endpoint uses the wrong summer offset.
query expect_fallback(Clock expressions with inputs that may be scalar require Spark for non-fixed timezones)
SELECT hour(IF(id > 0, timestamp_micros(9089364630896000000L), timestamp_micros(CAST(id AS BIGINT)))),
       minute(IF(id > 0, timestamp_micros(9089364630896000000L), timestamp_micros(CAST(id AS BIGINT)))),
       second(IF(id > 0, timestamp_micros(9089364630896000000L), timestamp_micros(CAST(id AS BIGINT))))
FROM scalar_clock_regional


statement
CREATE TABLE regional_clock_arrays(id INT, p BOOLEAN, ts1 TIMESTAMP, ts2 TIMESTAMP) USING parquet

statement
INSERT INTO regional_clock_arrays SELECT /*+ COALESCE(1) */ * FROM VALUES
  (1, true, TIMESTAMP '2024-07-01 12:34:56', TIMESTAMP '2024-01-01 01:02:03'),
  (2, true, CAST(NULL AS TIMESTAMP), TIMESTAMP '2024-01-01 04:05:06'),
  (3, false, TIMESTAMP '2024-07-01 07:08:09', TIMESTAMP '2024-01-01 10:11:12'),
  (4, false, CAST(NULL AS TIMESTAMP), CAST(NULL AS TIMESTAMP)) AS rows(id, p, ts1, ts2)

-- Every value branch is a column array, including homogeneous and all-NULL batches.
query expect_native(hour,minute,second)
SELECT id, hour(IF(p, ts1, ts2)), minute(IF(p, ts1, ts2)), second(IF(p, ts1, ts2))
FROM regional_clock_arrays

query expect_native(hour,minute,second)
SELECT id, hour(IF(p, ts1, ts2)), minute(IF(p, ts1, ts2)), second(IF(p, ts1, ts2))
FROM regional_clock_arrays WHERE id < 3

query expect_native(hour,minute,second)
SELECT id, hour(coalesce(ts1, ts2)), minute(coalesce(ts1, ts2)), second(coalesce(ts1, ts2))
FROM regional_clock_arrays

query expect_native(hour,minute,second)
SELECT id, hour(CASE WHEN p THEN ts1 ELSE ts2 END),
       minute(CASE WHEN p THEN ts1 ELSE ts2 END),
       second(CASE WHEN p THEN ts1 ELSE ts2 END)
FROM regional_clock_arrays

query expect_native(hour,minute,second)
SELECT id, hour(IF(p, coalesce(ts1, ts2), CASE WHEN id = 3 THEN ts1 ELSE ts2 END)),
       minute(IF(p, coalesce(ts1, ts2), CASE WHEN id = 3 THEN ts1 ELSE ts2 END)),
       second(IF(p, coalesce(ts1, ts2), CASE WHEN id = 3 THEN ts1 ELSE ts2 END))
FROM regional_clock_arrays

-- A scalar value branch still requires compatibility fallback, even with column predicates.
query expect_fallback(Clock expressions with inputs that may be scalar require Spark for non-fixed timezones)
SELECT hour(IF(p, ts1, TIMESTAMP '2200-07-01 12:34:56')) FROM regional_clock_arrays

query expect_fallback(Clock expressions with inputs that may be scalar require Spark for non-fixed timezones)
SELECT hour(CASE WHEN p THEN ts1 ELSE (SELECT max(ts2) FROM regional_clock_arrays) END)
FROM regional_clock_arrays

query expect_fallback(Clock expressions with inputs that may be scalar require Spark for non-fixed timezones)
SELECT hour(coalesce(ts1, TIMESTAMP '2200-07-01 12:34:56')) FROM regional_clock_arrays

-- A CASE without ELSE can synthesize a scalar NULL when no row matches.
query expect_fallback(Clock expressions with inputs that may be scalar require Spark for non-fixed timezones)
SELECT hour(CASE WHEN id < 0 THEN ts1 END) FROM regional_clock_arrays
