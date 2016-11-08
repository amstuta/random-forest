name := """play-getting-started"""

version := "1.0-SNAPSHOT"

scalaVersion := "2.10.6"

// libraryDependencies ++= Seq(
// )

libraryDependencies <+= scalaVersion("org.scala-lang" % "scala-compiler" % _ )
