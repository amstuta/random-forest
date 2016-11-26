package com.ensemble

import scala.util.Random
import scala.collection.mutable.{Map => MMap}
import com.tree.{DecisionTreeClassifier, Types}


class RandomForestClassifier(protected val nbTrees: Int,
                            protected val nbSamples: Int,
                            protected val maxDepth: Int = -1) {

  protected val trees = MMap[Int, DecisionTreeClassifier]()

  /**
   * Trains a random forest classifier using the iven dataset.
   *
   * @param features  List of Lists containing the input variables
   * @param targets   List of outputs corresponding to the given features
   */
  def fit(features: Types.Features, targets: List[String]) = {
    0 to nbTrees foreach { i =>
      println(s"Training tree ${i}")

      val randomSet = Random.shuffle(features.zip(targets)).take(nbSamples)
      val randomFeatures = randomSet map {tuple => tuple._1}
      val randomTargets = randomSet map {tuple => tuple._2}

      val tree = new DecisionTreeClassifier(maxDepth=maxDepth)
      tree.fit(randomFeatures, randomTargets)

      trees(i) = tree
    }
  }

  /**
   * Makes a prediction using the previously trained trees and the given
   * feature. It uses the "max votes" technique.
   *
   * @param feature A List containing the feature to make a prediction for
   * @return        The prediction (String)
   */
  def predict(feature: List[Any]) = {
    trees
      .values
      .map { tree => tree.predict(feature) }
      .groupBy(identity)
      .maxBy(_._2.size)
      ._1
  }

}
