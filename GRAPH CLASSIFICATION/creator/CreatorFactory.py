import logging

from .ModelCreator import ModelCreator


class CreatorFactory:

    def __init__(self):
        pass

    def get_creator(args, training_data, test_data):
        if args.creator == 'Model':
            return ModelCreator(args, training_data, test_data)
        else:
            logging.error('No such creator name as ' + args.creator)
            raise ValueError('Define the Creator')
