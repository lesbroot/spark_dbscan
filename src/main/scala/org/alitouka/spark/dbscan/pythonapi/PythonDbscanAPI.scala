package org.apache.spark.mllib.api.python

import org.apache.spark.annotation.DeveloperApi
import org.alitouka.spark.dbscan.{DbscanSettings, Dbscan, DbscanModel}
import org.alitouka.spark.dbscan.spatial.Point
import org.apache.spark.api.java.JavaRDD
import net.razorvine.pickle.Unpickler
import scala.collection.JavaConverters._
import java.util.ArrayList


class WrappedModel(model: DbscanModel) {

  def predict(data: Array[Byte]) = {
    val point = new Point(SerDe.deserializeDoubleVector(data).toArray)
    model.predict(point)
  }
}

@DeveloperApi
class PythonDbscanAPI extends Serializable {
  
  def train(inputData: JavaRDD[Array[Byte]], epsilon: Int = 25, numOfPoints: Int = 30) = {
    val clusteringSettings = new DbscanSettings().withEpsilon(epsilon).withNumberOfPoints(numOfPoints)
    val pointRDD = inputData.rdd.map(bytes => {
      new Point(SerDe.deserializeDoubleVector(bytes).toArray)
    })
    val model = Dbscan.train(pointRDD, clusteringSettings)
    new WrappedModel(model)
  }
}
