package com.tree

import scala.util.Random


/* Types used in this package */
package object Types {
  type Features = List[List[Any]]
}


/* Math functions used by trees */
private[tree] object MathUtil {
  import scala.math.log

  def log2(x: Double) = log(x) / log(2)
}


/**
 * Configuration of a decision tree.
 *
 * @param criteria  The criteria used to split: "entropy" or "gini"
 * @param maxDepth  If set, sets a limit of depth for the tree
 */
private[tree] case class TreeParameters(
  criterion: String,
  maxDepth: Int,
  randomFeatures: Boolean
)


/**
 * Represents a node in the tree.
 *
 * @param col         Column concerned in the features
 * @param value       Value used to match input
 * @param results     Result stored in the terminal nodes
 * @param trueBranch  Branch followed if input is matched
 * @param falseBranch Branch followed if input isn't matched
 */
sealed case class DecisionNode(
  col: Int = -1,
  value: Any = None,
  results: Option[Map[Int, Int]] = None,
  trueBranch: Option[DecisionNode] = None,
  falseBranch: Option[DecisionNode] = None
)


/**
 * Represents the splits used when building the tree.
 *
 * @param gain      Information gain while splitting a node
 * @param criteria  Column of data in this node and value in the node
 * @param set1      Data used in the following node in the true branch
 * @param set2      Data used in the following node in the false branch
 */
sealed case class Split(
  gain: Double = 0.0,
  criteria: Option[Tuple2[Int, Any]] = None,
  set1: Option[Tuple2[Types.Features, List[Int]]] = None,
  set2: Option[Tuple2[Types.Features, List[Int]]] = None
)


/**
 * Decision tree used for classification. Accepts any number of inputs and
 * predicts a class (represented by a string).
 */
class DecisionTreeClassifier(criterion: String = "entropy",
                             maxDepth: Int = -1,
                             randomFeatures: Boolean = false)
 extends TreeParameters(criterion, maxDepth, randomFeatures) {


  private var rootNode: Option[DecisionNode] = None
  private var targetsHashes: Map[Any, Int] = Map.empty
  private var featuresIndexes: List[Int] = List.empty


  private def splitter(x: List[Int]) = criterion match {
    case "entropy" => entropy(x)
    case "gini"    => gini(x)
    case _         => throw new IllegalArgumentException(s"Unknown criterion '${criterion}'")
  }


  /**
   * Fits the model using the features and targets given in parameter.
   *
   * @param features  A List of List with each row representing
   *                  the set of inputs for a certain target value
   * @param targets   List containing the target values matching the inputs
   */
  def fit(features: Types.Features, targets: List[String]) = {
    if (features.isEmpty) {
      throw new IllegalArgumentException("Empty list of features")
    }

    targetsHashes = hashTargets(targets)
    val intTargs = targets map { t => targetsHashes(t) }

    randomFeatures match {
      case true  =>
        featuresIndexes = chooseFeaturesIndexes(features(0))
        val randFeatures = features map { f => filterFeatures(f) }
        rootNode = Some(buildTree(randFeatures, intTargs, maxDepth))

      case false =>
        rootNode = Some(buildTree(features, intTargs, maxDepth))
    }
  }


  /**
   * Predicts a class given a set of inputs.
   *
   * @param feat  The List to predict a class for
   * @return      A Map[String, Int] containing the
   *              predicted class as the only key
   */
  def predict(feat: List[Any]) =
    rootNode match {
      case Some(s)  =>
        val predicted = randomFeatures match {
          case true  => classify(filterFeatures(feat), s)
          case false => classify(feat, s)
        }

        targetFromHash(predicted) match {
          case Some(r) => r
          case None    => throw new NoSuchFieldException("Internal error")
        }
      case None     => throw new IllegalStateException("Model not trained")
    }


  /**
   * Randomly chooses the indexes of the features to use in this tree.
   * @param feat  A feature set
   * @return      A List containing the indexes to use
   * TODO: make size of sample configurable
   */
  private def chooseFeaturesIndexes(feat: List[Any]) = {
    Random.shuffle(0 to feat.length).take(feat.length / 2).toList
  }


  /**
   * Returns a list containing only the features at the indexes contained
   * in featuresIndexes.
   * @param feat  A feature set
   * @return      A subset of the original features
   */
  private def filterFeatures(feat: List[Any]) = {
    featuresIndexes map { i => feat(i) }
  }


  /**
   * Classifies the input by traversing the tree.
   *
   * @param feat  The input to predict a class for
   * @param node  The current node in the tree
   */
  private def classify(feat: List[Any], node: DecisionNode): Int =
    node.results match {
      case Some(s) => s.head._1
      case None    =>
        val v = feat(node.col)
        val branch = v match {
          case _:Int | _:Double =>
            if (v.toString.toDouble >= node.value.toString.toDouble) node.trueBranch.get
            else node.falseBranch.get
          case _ =>
            if (v == node.value) node.trueBranch.get else node.falseBranch.get
        }
        classify(feat, branch)
    }


  /**
   * Builds the nodes of the tree by splitting input data until no gain
   * of information is added.
   */
  private def buildTree(feats: Types.Features, targets: List[Int], depth: Int): DecisionNode =
    feats.length match {
      case 0 => DecisionNode()
      case _ =>
        depth match {
          case 0 => DecisionNode(results=Some(countResults(targets)))
          case _ =>
            val split = bestSplit(feats, targets)

            split.gain match {
              case 0 => DecisionNode(results=Some(countResults(targets)))
              case _ =>
                val (trueF, trueT) = split.set1.get
                val (falseF, falseT) = split.set2.get
                val (col, value) = split.criteria.get

                val trueBranch = buildTree(trueF, trueT, depth - 1)
                val falseBranch = buildTree(falseF, falseT, depth - 1)

                DecisionNode(col, value, None, Some(trueBranch), Some(falseBranch))
            }
        }
    }


  /**
   * Finds the best variable to split the data in the current node.
   */
  private def bestSplit(feats: Types.Features, targets: List[Int]) = {

    val curScore = splitter(targets)
    var bestSplit = Split()

    0 to (feats(0).length - 1) foreach { col =>
      val values = countResults(feats map { f => f(col) })

      values.keys foreach { value =>
        val (s1, t1, s2, t2) = divideSet(feats, targets, col, value)
        val p = s1.length.toDouble / feats.length
        val gain = curScore - p * splitter(t1) - (1.0 - p) * splitter(t2)

        if (gain > bestSplit.gain && s1.length > 0 && s2.length > 0)
          bestSplit = Split(gain, Some((col, value)), Some((s1, t1)), Some((s2, t2)))
      }
    }
    bestSplit
  }


  /**
   * Divides the input data of a node.
   */
  private def divideSet(feats: Types.Features, targs: List[Int],
                        col: Int, value: Any) = {

    val split = value match {
      case _:Int | _:Double =>
        (row: List[Any]) => row(col).toString.toDouble >= value.toString.toDouble
      case _                =>
        (row: List[Any]) => row(col) == value
    }

    val set1 = feats zip targs filter { case (f,t) => split(f) }
    val set2 = feats zip targs filter { case (f,t) => !split(f) }

    (set1.map(_._1), set1.map(_._2), set2.map(_._1), set2.map(_._2))
  }


  /**
   * Calculates the entropy in the given dataset.
   */
  private def entropy[T <% Ordered[T]](targs: List[T]) = {
    val res = countResults(targs)
    res.keys.foldLeft(0.0) {
      case (ent, key) =>
        val p = res(key).asInstanceOf[Double] / targs.length
        ent - p * MathUtil.log2(p)
    }
  }


  /**
   * Calculates the gini coefficient in the given dataset.
   */
  def gini[T <% Ordered[T]](targs: List[T]) = {
   val (height, area) = targs.sorted.foldLeft((0.0, 0.0)) {
     case (vals, key) =>
       val value = key.asInstanceOf[Int]
       (vals._1 + value, vals._2 + (vals._1 + value - value / 2.0))
   }
   val fairArea = height * targs.length / 2.0
   (fairArea - area) / fairArea
 }


  /**
   * Counts the occurence of each target in the given data.
   */
  private def countResults[T](targs: List[T]): Map[T, Int] =
    targs.groupBy(identity).mapValues(_.size)


  /**
   * Returns a Map containing the target values as keys and
   * their hashes as values.
   */
  private def hashTargets[T](targs: List[T]): Map[T, Int] =
    targs.distinct.map{ v => (v -> v.hashCode) }.toMap


  /**
   * Retrieve the target name from the hash stored
   */
  private def targetFromHash(hash: Int) = targetsHashes.map(_.swap) get hash


  override def toString = printTree(rootNode.get, "")


  private def printTree(node: DecisionNode, indent: String): String =
    node.results match {
      case None =>
        s"${node.col}: ${node.value}\n" +
        s"${indent} T -> ${printTree(node.trueBranch.get, indent + " ")}" +
        s"${indent} F -> ${printTree(node.falseBranch.get, indent + " ")}"
      case _    =>
        s"${targetFromHash(node.results.get.head._1).get}\n"
    }

}
