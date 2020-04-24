



def load_dataset(config):
    pass


if __name__ == "__main__":
    import sys, os
    abs_path = os.path.abspath('./')
    # print(abs_path)
    sys.path.append(abs_path)
    from common import load_config
    config = load_config(
        '/Users/faith/Desktop/pytorch/config/example.policy.yml')
    print(config)