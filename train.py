import numpy as np
import tensorflow as tf
from _utils.generate import Generator
from transformer_xl import TransformerXL
from _utils.utils import decode_fn, char_recursion, WarmUpCosineDecayScheduler
from config import configure as cfg

if __name__ == '__main__':

    data_gen = Generator(verbose=cfg.verbose,
                         dataset=cfg.dataset,
                         file_dir=cfg.file_dir,
                         batch_size=cfg.batch_size,
                         sequence_len=cfg.sequence_len)

    n_token = data_gen.vocabulary.__len__()

    transformer = TransformerXL(n_head=cfg.n_head,
                                d_head=cfg.d_head,
                                d_embed=cfg.d_embed,
                                n_token=n_token,
                                cutoffs=cfg.cutoffs,
                                tie_projs=cfg.tie_projs,
                                batch_size=cfg.batch_size,
                                embed_size=cfg.embed_size,
                                divide_rate=cfg.divide_rate,
                                num_layers=cfg.num_layers,
                                member_len=cfg.member_len,
                                learning_rate=cfg.learning_rate)

    ckpt = tf.train.Checkpoint(transformer=transformer.model,
                               optimizer=transformer.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_info, valid_info = data_gen.generate_tfrecords()
    train_records_path = train_info["train_file_path"]
    train_steps = train_info["train_steps"]
    train_set = tf.data.TFRecordDataset([train_records_path])
    train_set = train_set.map(decode_fn).cache().repeat()
    train_set = train_set.batch(cfg.batch_size, drop_remainder=True)

    if valid_info is not None:
        valid_records_path = valid_info["valid_file_path"]
        valid_steps = valid_info["valid_steps"]
        valid_set = tf.data.TFRecordDataset([valid_records_path])
        valid_set = valid_set.map(decode_fn).cache().repeat()
        valid_set = valid_set.batch(cfg.batch_size, drop_remainder=True)

    if cfg.cosine_scheduler:
        total_steps = train_steps * cfg.Epoches
        warmup_steps = int(train_steps * cfg.Epoches * 0.2)
        hold_steps = train_steps * data_gen.batch_size
        reduce_lr = WarmUpCosineDecayScheduler(global_interval_steps=total_steps,
                                               warmup_interval_steps=warmup_steps,
                                               hold_interval_steps=hold_steps,
                                               learning_rate_base=cfg.learning_rate,
                                               warmup_learning_rate=cfg.warmup_learning_rate,
                                               min_learning_rate=cfg.min_learning_rate,
                                               verbose=0)

    for i in range(cfg.Epoches):
        for j, batch_data in enumerate(train_set):
            src_indices = batch_data['source'].indices
            tgt_indices = batch_data['target'].indices
            sources = batch_data['source'].values
            targets = batch_data['target'].values

            sources = data_gen.assign_gathered(sources, src_indices)
            targets = data_gen.assign_gathered(targets, tgt_indices)
            if cfg.cosine_scheduler:
                learning_rate = reduce_lr.batch_begin()
                transformer.optimizer.learning_rate = learning_rate
            transformer.train(sources, targets)

            if not (j + 1) % cfg.per_sample_interval:
                pred_indices = transformer.generate_sample(sources, j + 1)
                orig_tokens = np.array(data_gen.vocabulary)[sources]
                pred_tokens = np.array(data_gen.vocabulary)[pred_indices]
                for orig_token, pred_token in zip(orig_tokens, pred_tokens):
                    former_sentence = char_recursion(orig_token.tolist())
                    latter_sentence = char_recursion(pred_token.tolist())
                    print("steps:{:d}-result:{:s}".format((j + 1), former_sentence + latter_sentence))

            if (j + 1) == train_steps:
                break

        transformer.members = [np.random.randn(cfg.batch_size, 0, cfg.embed_size)
                               for i in range(cfg.batch_size)]
        print(f'Epoch {i + 1}, '
              f'train_loss: {transformer.train_loss.result()}, '
              f'train_acc: {transformer.train_acc.result() * 100}')

        ckpt_save_path = ckpt_manager.save()

        if valid_info is not None:
            for j, batch_data in enumerate(valid_set):
                src_indices = batch_data['source'].indices
                tgt_indices = batch_data['target'].indices
                sources = batch_data['source'].values
                targets = batch_data['target'].values

                sources = data_gen.assign_gathered(sources, src_indices)
                targets = data_gen.assign_gathered(targets, tgt_indices)

                transformer.validate(sources, targets)

                if (j + 1) == valid_steps:
                    break

            transformer.members = [np.random.randn(cfg.batch_size, 0, cfg.embed_size)
                                   for i in range(cfg.batch_size)]
            print(f'Epoch {i + 1}, '
                  f'valid_loss: {transformer.valid_loss.result()}, '
                  f'valid_acc: {transformer.valid_acc.result() * 100}')

        transformer.train_acc.reset_states()
        transformer.valid_acc.reset_states()
        transformer.train_loss.reset_states()
        transformer.valid_loss.reset_states()
        transformer.train_f1_score.reset_states()
        transformer.valid_f1_score.reset_states()
