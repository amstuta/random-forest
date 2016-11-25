import com.tree.DecisionTreeClassifier
import com.ensemble.RandomForestClassifier
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.{SQLContext, DataFrame, SparkSession}


case class Data(
  _c0: Int,
  _c1: String,
  _c2: String,
  _c3: Int,
  _c4: String,
  _c5: String,
  _c6: String,
  _c7: String,
  _c8: String,
  _c9: Int,
  _c10: Int,
  _c11: Int,
  _c12: String,
  _c13: String
)


object Main {

  def test(spark: SparkSession, df: DataFrame) = {
    import spark.implicits._

    df.show()
    df.printSchema()

    val ds = df.as[(Int, String, String, Int, String, String, String, String,
                    String, Int, Int, Int, String, String)]

    ds.foreach { row =>
      row.productIterator.foreach(println)
    }
  }

  def main(args: Array[String]) {
    val spark = SparkSession
                  .builder()
                  .appName("Random Forest")
                  .master("local[*]")
                  .getOrCreate()

    val df = spark
              .read
              .option("header", "false")
              .option("inferSchema", "true")
              .csv("data/income.csv")

    test(spark, df)

    spark.stop
  }

}
