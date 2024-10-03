import argparse

class Parser:

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='GDSS')
        self.parser.add_argument('--type', type=str, required=True)

        self.set_arguments()

    def set_arguments(self):

        self.parser.add_argument('--config', type=str,
                                    required=True, help="Path of config file")
        self.parser.add_argument('--reward_name', type=str, default="Reward",
                                 required=False, help="reward name for plot")
        self.parser.add_argument('--train_strat', type=str, default="rewd",
                                 required=False, help="ori or rewd or PM")
        self.parser.add_argument('--comment', type=str, default="", 
                                    help="A single line comment for the experiment")
        self.parser.add_argument('--sample_M', type=int, default=20)
        self.parser.add_argument('--alpha', type=float, default=0.3) 
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--version', type=int, default = 1)
        

    def parse(self):

        args, unparsed  = self.parser.parse_known_args()
        
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        
        return args