import json
import protocol as protocol

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as config_f:
        config = json.load(config_f)
    protocol_name = config["protocol"]
    md_protocol = getattr(protocol, f'{protocol_name}Protocol')(config)

    # load and prepare system

    md_protocol.run_protocol()
    md_protocol.post_process()
