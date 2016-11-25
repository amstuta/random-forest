name := """Random Forest"""

version := "1.0-SNAPSHOT"

scalaVersion := "2.10.6"

libraryDependencies <+= scalaVersion("org.scala-lang" % "scala-compiler" % _ )

libraryDependencies ++= Seq(
  "org.apache.spark"  %%  "spark-core"      % "2.0.2",
  "org.apache.spark"  %   "spark-sql_2.10"  % "2.0.2"
)
