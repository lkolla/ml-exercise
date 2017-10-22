package ml.regression

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object SimpleLinearRegression {

  def main(args: Array[String]): Unit = {

    val session = SparkSession.builder().appName("SimpleLinearRegression").master("local[*]").getOrCreate()

    val training = session.read.format("libsvm").load("src/main/resources/sample_linear_regression_data.txt")

    val lr = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)

    val model = lr.fit(training)

    println(s"Coefficients: ${model.coefficients} \t Intercepts ${model.intercept}")

    val trainingSummary = model.summary

    println(s"number of iterations: ${trainingSummary.totalIterations}")
    println(s"objective history: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show(50)
    println(s"SSE: ${trainingSummary.r2}")
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  }


}
