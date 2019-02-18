
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.Interaction
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

// Original correlation

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val df = sqlContext.read.format("jdbc").option("url", "jdbc:mysql://13.56.49.220:3306/beast_mode_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "training_set").option("user", "bmq_user").option("password", "Be@stM0de").load()
//df.show()


val assembler1 = new VectorAssembler().setInputCols(Array( "fatigue_index","intensity_index","bmq_index")).setOutputCol("features").transform(df)
//assembler1.show()

val Row(coeff1: Matrix) = Correlation.corr(assembler1, "features").head
println("Pearson correlation matrix:\n" + coeff1.toString)

val Row(coeff2: Matrix) = Correlation.corr(assembler1, "features", "spearman").head
println("Spearman correlation matrix:\n" + coeff2.toString)


// Sleep correlation


val df_sleep = sqlContext.read.format("jdbc").option("url", "jdbc:mysql://13.56.49.220:3306/beast_mode_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "(select DATE_FORMAT(date(diary_day), \"%Y-%m-%d\") as DIARY_DAY, TOTAL_MINUTES_ASLEEP, avg(BMQ) as BMQ from SLEEP_HISTORY, BMQ_HISTORY where DATE_FORMAT(date(diary_day), \"%Y-%m-%d\") = DATE_FORMAT(date(BMQ_HISTORY.TIME_ENTERED), \"%Y-%m-%d\") group by DATE_FORMAT(date(diary_day), \"%Y-%m-%d\"), TOTAL_MINUTES_ASLEEP order by DATE_FORMAT(date(diary_day), \"%Y-%m-%d\"), TOTAL_MINUTES_ASLEEP ) tmp").option("user", "bmq_user").option("password", "Be@stM0de").load()
//df_sleep.show()

val assembler_sleep = new VectorAssembler().setInputCols(Array( "TOTAL_MINUTES_ASLEEP","BMQ")).setOutputCol("features").transform(df_sleep)
//assembler_sleep.show()

val Row(coeff1: Matrix) = Correlation.corr(assembler_sleep, "features").head
println("Pearson correlation matrix:\n" + coeff1.toString)

val Row(coeff2: Matrix) = Correlation.corr(assembler_sleep, "features", "spearman").head
println("Spearman correlation matrix:\n" + coeff2.toString)


// HR correlation


val df_hr = sqlContext.read.format("jdbc").option("url", "jdbc:mysql://13.56.49.220:3306/beast_mode_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "(select DATE_FORMAT(date(diary_day), \"%Y-%m-%d\") as DIARY_DAY, REST_HR, avg(BMQ) as BMQ from HEALTH_HISTORY, BMQ_HISTORY where DATE_FORMAT(date(diary_day), \"%Y-%m-%d\") = DATE_FORMAT(date(BMQ_HISTORY.TIME_ENTERED), \"%Y-%m-%d\") group by DATE_FORMAT(date(diary_day), \"%Y-%m-%d\"), REST_HR order by DATE_FORMAT(date(diary_day), \"%Y-%m-%d\"), REST_HR ) tmp").option("user", "bmq_user").option("password", "Be@stM0de").load()
//df_hr.show()

val assembler_hr = new VectorAssembler().setInputCols(Array( "REST_HR","BMQ")).setOutputCol("features").transform(df_hr)
//assembler_hr.show()

val Row(coeff1: Matrix) = Correlation.corr(assembler_hr, "features").head
println("Pearson correlation matrix:\n" + coeff1.toString)

val Row(coeff2: Matrix) = Correlation.corr(assembler_hr, "features", "spearman").head
println("Spearman correlation matrix:\n" + coeff2.toString)