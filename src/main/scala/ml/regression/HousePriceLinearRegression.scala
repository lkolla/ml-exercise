package ml.regression

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.SparkSession



object HousePriceLinearRegression {

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
    printModel(lr.fit(algInput))

  }

  def printModel(model: LinearRegressionModel) = {
    println(s"number of iterations: ${model.summary.totalIterations}")
    println(s"objective history: ${model.summary.objectiveHistory.toList}")
    println(s"SSE: ${model.summary.r2}")
    println(s"RMSE: ${model.summary.rootMeanSquaredError}")
    println(s"MAE: ${model.summary.meanAbsoluteError}")
    println(s"devianceResiduals: ${model.summary.devianceResiduals.toList}")
    println(s"MAE: ${model.summary.explainedVariance}")
    println(s"coefficientStandardErrors: ${model.summary.coefficientStandardErrors.toList}")

    model.summary.residuals.show(50)
    model.summary.predictions.show(50)
  }

}
