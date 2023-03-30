import sentence_transformers
import os
import argparse
import torchvision.models.resnet


def add_arguments(parser):
    parser.add_argument('--language_model_name', type=str, default='bert-base-nli-mean-tokens', help='Name of the language model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')


def main(args):
    os.environ['USE_TORCH'] = '1'
    language_model = sentence_transformers.SentenceTransformer(args.language_model_name, device=args.device, cache_folder='./cache/')
    print(language_model.encode(['This is an example sentence', 'Each sentence is converted']))

    vision_model = torchvision.models.resnet.resnet18(pretrained=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
