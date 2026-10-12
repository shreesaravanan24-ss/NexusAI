"""Environment and Spark Session Setup Utility for NexusAI.

Ensures that JAVA_HOME and HADOOP_HOME point to compatible binaries
(supporting Windows local executions and JDK 17 LTS).
"""

from __future__ import annotations

import os
from pathlib import Path
from pyspark.sql import SparkSession


def configure_environment() -> None:
    """Configures JAVA_HOME and HADOOP_HOME if bundled locally or not set."""
    base_dir = Path(__file__).resolve().parent.parent
    tools_dir = base_dir / ".tools"
    local_jdk = tools_dir / "jdk-17"
    local_hadoop = tools_dir / "hadoop"

    # Use bundled JDK 17 if available or if system Java is incompatible
    if local_jdk.exists() and (
        "JAVA_HOME" not in os.environ or not os.environ["JAVA_HOME"]
    ):
        os.environ["JAVA_HOME"] = str(local_jdk)
        bin_path = str(local_jdk / "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_path};{os.environ.get('PATH', '')}"

    # Use bundled Windows Hadoop binaries (winutils.exe & hadoop.dll)
    if local_hadoop.exists() and (
        "HADOOP_HOME" not in os.environ or not os.environ["HADOOP_HOME"]
    ):
        os.environ["HADOOP_HOME"] = str(local_hadoop)
        bin_path = str(local_hadoop / "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_path};{os.environ.get('PATH', '')}"


def get_spark_session(app_name: str = "NexusAI-ETL") -> SparkSession:
    """Builds and returns a tuned, local-first SparkSession."""
    configure_environment()

    spark = (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
