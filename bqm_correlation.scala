
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.Interaction
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row


val sqlContext = new org.apache.spark.sql.SQLContext(sc)
val df = sqlContext.read.format("jdbc").option("url", "jdbc:mysql://13.56.49.220:3306/beast_mode_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "training_set").option("user", "bmq_user").option("password", "Be@stM0de").load()
df.show()


val assembler1 = new VectorAssembler().setInputCols(Array( "fatigue_index","intensity_index","bmq_index")).setOutputCol("features").transform(df)
assembler1.show()


val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(2.0).transform(assembler1)
normalizer.show()

val Row(coeff1: Matrix) = Correlation.corr(normalizer, "normFeatures").head
println("Pearson correlation matrix:\n" + coeff1.toString)