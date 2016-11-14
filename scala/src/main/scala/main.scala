import io.Source
import com.tree.DecisionTreeClassifier
import com.ensemble.RandomForestClassifier


object Main {

  def main(args: Array[String]) {

    val dataset = Source.fromFile("data/income.csv")
                    .getLines
                    .map { line => line.split(",").map(_.trim) }
                    .toList

    val (train, test) = dataset splitAt 22793
    val trFeatures = train.map { data => data.dropRight(1).toList }.filter(!_.isEmpty)
    val teFeatures = test.map { data => data.dropRight(1).toList }.filter(!_.isEmpty)
    val trTargets = train.map { data => data.last }.filter(!_.isEmpty)
    val teTargets = test.map { data => data.last }.filter(!_.isEmpty)

    val rf = new RandomForestClassifier(30, 3000)
    rf.fit(trFeatures, trTargets)

    // val tree = new DecisionTreeClassifier(maxDepth=20)
    // tree.fit(trFeatures, trTargets)

    val error = teFeatures.zip(teTargets).map {
      case (f,t) =>
        val predicted = rf.predict(f) //tree.predict(f)
        println(s"${f}, ${t} => ${predicted}")
        println(predicted)
        t == predicted
    }.filter(!_).length

    // println(tree)

    println("************")
    println(s"Error rate: ${error.toDouble / teTargets.length * 100}")
    println("************")

  }

}
