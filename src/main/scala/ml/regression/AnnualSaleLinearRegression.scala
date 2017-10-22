package ml.regression

import ml.regression.utils.Util
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

/**
  * this `CustomerPurchaseDataLinearRegression` perform [[LinearRegression]] on '''e-commerce.csv''' data.
  */
object AnnualSaleLinearRegression {

  /**
    * this `main` method read '''e-commerce.csv''' data and perform [[LinearRegression]] operation to find out
    * difference between '''Annual spending prediction vs actual '''
    * @param args
    */
  def main(args: Array[String]): Unit = {

    //initialize the spark session with default settings.
    val session = SparkSession
      .builder()
      .appName("AnnualSaleLinearRegression")
      .master("local[*]")
      .getOrCreate()

    import session.implicits._

    val customerDF = session.read.option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("src/main/resources/e-commerce.csv")

    //for debugging.
    customerDF.printSchema()
    customerDF.show()
    customerDF.columns.foreach(println)

    //merging the feature columns to features Vector.
    val featuresBuilder = new VectorAssembler()
      .setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership"))
      .setOutputCol("features")


    val inputDF = featuresBuilder.transform(customerDF.select($"Yearly Amount Spent".as("label"), $"Avg Session Length",
      $"Time on App", $"Time on Website", $"Length of Membership"))
      .select($"label", $"features")

    //for debugging.
    inputDF.printSchema()
    inputDF.show(5)


    val linearReg = new LinearRegression()

    val summary = linearReg.fit(inputDF)

    Util.printModel(summary, session)


    //stop the spark application.
    session.stop()
  }


}
