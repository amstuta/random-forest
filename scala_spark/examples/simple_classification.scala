import com.tree.DecisionTreeClassifier

object Main {

  def main(args: Array[String]) {

    val data = List(List("slashdot", "USA", "yes", 18),
                    List("google", "France", "yes", 23),
                    List("digg", "USA", "yes", 24),
                    List("kiwitobes", "France", "yes", 23),
                    List("google", "UK", "no", 21),
                    List("(direct)", "New Zealand", "no", 12),
                    List("(direct)", "UK", "no", 21),
                    List("google", "USA", "no", 24),
                    List("slashdot", "France", "yes", 19),
                    List("digg", "USA", "no", 18),
                    List("google", "UK", "no", 18),
                    List("kiwitobes", "UK", "no", 19),
                    List("digg", "New Zealand", "yes", 12),
                    List("slashdot", "UK", "no", 21),
                    List("google", "UK", "yes", 18),
                    List("kiwitobes", "France", "yes", 19))

    val targets = List("None", "Premium", "Basic", "Basic", "Premium", "None",
                        "Basic", "Premium", "None", "None", "None", "None",
                        "Basic", "None", "Basic", "Basic")


    val tree = new DecisionTreeClassifier
    tree.fit(data, targets)

    println(tree.predict(List("(direct)", "USA", "yes", 5)))
    println(tree.predict(List("(direct)", "USA", "no", 23)))
    println(tree.predict(List("google", "France", "no", 21)))

    println(tree)
  }
}
