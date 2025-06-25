from chestCancerClassifier import logger
from chestCancerClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from chestCancerClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from chestCancerClassifier.pipeline.stage_03_training import TrainingPipeline
from chestCancerClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

STAGE_NAME = "Data Ingestion stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.run()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Prepare Base Model stage"

if __name__ == "__main__":
    try:
        logger.info("********************************************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.run()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training stage"

if __name__ == "__main__":
    try:
        logger.info("********************************************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.run()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation stage"

if __name__ == "__main__":
    try:
        logger.info("********************************************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.run()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
