package ml.regression

import ml.regression.utils.Util
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession


/**
  * this `HousePriceLinearRegression` perform [[LinearRegression]] on '''housing-data.csv''' data.
  */
object HousePriceLinearRegression {

  /**
    * this `main` method read '''housing-data.csv''' data and perform [[LinearRegression]] operation to find out
    * difference between '''House price prediction vs actual '''
    * @param args
    */
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession
      .builder()
      .appName("houses-price-linear-regression")
      .master("local[*]")
      .getOrCreate()

    import sparkSession.implicits._

    val houseDF = sparkSession.read
      .option("header","true")
      .option("inferSchema","true")
      .format("csv")
      .load("src/main/resources/housing-data.csv")

    houseDF.columns.foreach(println)

    //Setting up the data for ML..
    // label -> price
    // features -> remaining columns

    val intermidateDF = houseDF.select($"Price".as("label"), $"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms",
      $"Avg Area Number of Bedrooms", $"Area Population")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms", "Avg Area Number of Bedrooms", "Area Population"))
        .setOutputCol("features")


    val algInput = assembler.transform(intermidateDF).select($"label", $"features")

    val lr = new LinearRegression()
    Util.printModel(lr.fit(algInput), sparkSession)

  }

}
