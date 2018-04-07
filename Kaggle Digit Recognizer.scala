import org.apache.spark._
//import org.apache.spark.h2o.RDD
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import scala.io.Source

object TrainRFMLLib {


  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Kaggle - TrainRFMLLib")
    val sc = new SparkContext(conf)

    val binarized = sc.textFile("/FileStore/tables/train.csv")

    val labeled = binarized
      .map(_.split(",").map(_.toDouble))
      .map{values =>
        LabeledPoint(values.head, Vectors.dense(values.tail))
      }

    val categoricalFeaturesInfo =
      (for(i <- 0 to 783) yield (i,2)).toMap

    val foldSize = 42000 / 5

    val testNbTree = Seq(10, 50, 100, 200, 300, 500)
    val testMaxDepth = Seq(5, 10, 20, 30)

    for {
      nbTree <- testNbTree
      maxDepth <- testMaxDepth
    }  {

      val scores = for (i <- 1 to 5) yield {
        val split =
          labeled.zipWithIndex().map {
            case (row, idx) if idx >= (i - 1) * foldSize && idx < i * foldSize =>
              (row, true)
            case (row, idx) =>
              (row, false)
          }
        val training = split.filter(_._2 == false).map(_._1)
        val testing = split.filter(_._2 == true).map(_._1)

        val model = RandomForest.trainClassifier(
          input = training,
          numClasses = 10,
          categoricalFeaturesInfo = categoricalFeaturesInfo,
          numTrees = nbTree,
          featureSubsetStrategy = "auto",
          impurity = "gini",
          maxDepth = maxDepth,
          maxBins = 100,
          seed = 199)

        computeAndPrintMetrics(s"nbTree=$nbTree maxDepth=$maxDepth k=$i", model, testing)
      }

      val total = scores.reduce((acc,value) => (acc._1 + value._1, acc._2 + value._2))
      println(s"********=> nbTree=$nbTree, maxDepth=$maxDepth, meanPrecision=${total._1 / 5}, meanRecall=${total._2 / 5}")
    }

  }

  private def computeAndPrintMetrics(id:String, model:RandomForestModel, testingSet:RDD[LabeledPoint]):(Double,Double) = {
    val predictions = testingSet.map { item =>
      (model.predict(item.features), item.label)
    }
    val metrics = new MulticlassMetrics(predictions)
    println("*******************************")
    println(s"Metrics for $id")
    //println(metrics.confusionMatrix)
    //println(s"f1 : ${metrics.fMeasure}")
    val (totPrecision, totRecall) =
      (for (label <- 0 to 9) yield (metrics.precision(label), metrics.recall(label)))
      .reduceLeft((acc,value)=>(acc._1 + value._1, acc._2 + value._2))
    val (meanPrecision, meanRecall) = (totPrecision/10, totRecall / 10)
    println(s"Mean precision: $meanPrecision")
    println(s"Mean recall: $meanRecall")
    (meanPrecision, meanRecall)
  }


  private def makeTestPredictions(sc:SparkContext, model:RandomForestModel):Unit = {
    val testBinarize = sc.textFile("/FileStore/tables/train.csv")

    val predictions = testBinarize
      .zipWithIndex()
      .map {
        case (row, i) => (i, Vectors.dense(row.split(",").map(_.toDouble)))
      }
      .map{
        case (i, features) => (i, model.predict(features))
      }

    predictions
      .repartition(1)
      .sortBy(_._1)
      .map{
        case (i, predictedClass) => s"${i+1},${predictedClass.toInt}"
      }
      .saveAsTextFile("/tmp/predictions_RFMLLib.csv")
    predictions.take(100)
  
  }
}
