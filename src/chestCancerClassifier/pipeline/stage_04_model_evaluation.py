from chestCancerClassifier.components.model_evaluation import Evaluation
from chestCancerClassifier.config.configuration import ConfigurationManager
from chestCancerClassifier import logger

STAGE_NAME = "Model Evaluation stage"

class EvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            evaluation = Evaluation(eval_config)
            evaluation.evaluation()
            evaluation.log_into_mlflow()

        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.run()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e













