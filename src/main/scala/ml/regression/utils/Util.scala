package ml.regression.utils

import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession

object Util {
  def printModel(model: LinearRegressionModel, session: SparkSession) = {

    import session.implicits._


    println(s"number of iterations: ${model.summary.totalIterations}")
    println(s"objective history: ${model.summary.objectiveHistory.toList}")
    println(s"SSE: ${model.summary.r2}")
    println(s"RMSE: ${model.summary.rootMeanSquaredError}")
    println(s"MAE: ${model.summary.meanAbsoluteError}")
    println(s"devianceResiduals: ${model.summary.devianceResiduals.toList}")
    println(s"MAE: ${model.summary.explainedVariance}")
    println(s"coefficientStandardErrors: ${model.summary.coefficientStandardErrors.toList}")

    model.summary.predictions.select($"label", $"features", $"prediction", ($"label" - $"prediction").as("residuals")).show()
  }
}
