from chestCancerClassifier.components.training import Training
from chestCancerClassifier.config.configuration import ConfigurationManager
from chestCancerClassifier import logger

STAGE_NAME = "Training stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            config = ConfigurationManager()
            training_config = config.get_training_config()
            training = Training(config=training_config)
            training.get_base_model()
            training.train_valid_loader()
            training.train()

        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.run()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
