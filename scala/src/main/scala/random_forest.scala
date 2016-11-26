package com.ensemble

import scala.util.Random
import scala.collection.mutable.{Map => MMap}
import com.tree.{DecisionTreeClassifier, Types}


class RandomForestClassifier(protected val nbTrees: Int,
                            protected val nbSamples: Int,
                            protected val maxDepth: Int = -1) {

  protected val trees = MMap[Int, DecisionTreeClassifier]()


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


  def predict(feature: List[Any]) = {
    trees
      .values
      .map { tree => tree.predict(feature) }
      .groupBy(identity)
      .maxBy(_._2.size)
      ._1
  }

}
