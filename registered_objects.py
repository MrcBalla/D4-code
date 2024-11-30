import src.data.pipelines as ppls
import src.datatypes.features as features
import src.models as models
import src.test as test
import src.callbacks as clb


def main():
    print(ppls.reg_download)
    print(ppls.reg_preprocess)
    print(ppls.reg_runtime_t)
    print(features.reg_features)
    print(models.reg_models)
    print(models.reg_architectures)
    print(test.reg_metrics)
    print(test.reg_assignment)
    print(clb.reg_checkpoints)
    print(clb.reg_early_stopping)
    print(clb)


if __name__ == '__main__':
    main()