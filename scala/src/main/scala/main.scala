import io.Source
import scala.util.Random
import com.tree.DecisionTreeClassifier
import com.ensemble.RandomForestClassifier


object Main {

  def readDataset = {
    val dataset = Source.fromFile("../data/income.csv")
                    .getLines
                    .map { line => line.split(",").map(_.trim) }
                    .toList

    val nbSamples = 22793
    val shuffle = Random.shuffle(dataset)
    val (train, test) = shuffle splitAt nbSamples

    val trFeatures = train.map { data => data.dropRight(1).toList }.filter(!_.isEmpty)
    val teFeatures = test.map { data => data.dropRight(1).toList }.filter(!_.isEmpty)
    val trTargets = train.map { data => data.last }.filter(!_.isEmpty)
    val teTargets = test.map { data => data.last }.filter(!_.isEmpty)

    (trFeatures, teFeatures, trTargets, teTargets)
  }

  def testTree = {
    println("----------- Testing decision tree -----------")

    val (trainF, testF, trainT, testT) = readDataset
    val tree = new DecisionTreeClassifier(maxDepth=20)
    tree.fit(trainF, trainT)

    val error = testF.zip(testT).map {
      case (f,t) =>
        val predicted = tree.predict(f)
        t == predicted
    }.filter(!_).length

    println("************")
    println(s"Error rate: ${error.toDouble / testT.length * 100}")
    println("************")
  }

  def testRandomForest = {
    println("----------- Testing random forest -----------")

    val (trainF, testF, trainT, testT) = readDataset

    val rf = new RandomForestClassifier(30, 3000)
    rf.fit(trainF, trainT)

    val error = testF.zip(testT).map {
      case (f,t) =>
        val predicted = rf.predict(f)
        t == predicted
    }.filter(!_).length

    println("************")
    println(s"Error rate: ${error.toDouble / testT.length * 100}")
    println("************")
  }

  def main(args: Array[String]) {
    testTree
    testRandomForest
  }

}
