from .base_lightningmodule import BaseLightningModule

class LabelQueriedSeparationLightning(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}

        copylb = batch_data_dict['label_vector'].clone()
        if copylb.dim() == 3:
            assert copylb.shape[:2] == output_dict['waveform'].shape[:2]
        elif copylb.dim() == 2:
            assert copylb.shape[1] % output_dict['waveform'].shape[1] == 0
            copylb = copylb.view(
                copylb.shape[0],
                output_dict['waveform'].shape[1],
                copylb.shape[1] // output_dict['waveform'].shape[1])
        target_dict = {'waveform': batch_data_dict['dry_sources'],
                       'label_vector': copylb
                       }
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}

        copylb = batch_data_dict['label_vector'].clone()
        if copylb.dim() == 3:
            assert copylb.shape[:2] == output_dict['waveform'].shape[:2]
        elif copylb.dim() == 2:
            assert copylb.shape[1] % output_dict['waveform'].shape[1] == 0
            copylb = copylb.view(
                copylb.shape[0],
                output_dict['waveform'].shape[1],
                copylb.shape[1] // output_dict['waveform'].shape[1])
        target_dict = {'waveform': batch_data_dict['dry_sources'],
                       'label_vector': copylb
                       }
        # ── DEBUG ──────────────────────────────────────────────────────
        if batch_idx == 0:
            print(f"\n{'='*60}", flush=True)
            print(f"[DEBUG val] pred   power : {output_dict['waveform'].pow(2).mean().item():.6f}", flush=True)
            print(f"[DEBUG val] target power : {target_dict['waveform'].pow(2).mean().item():.6f}", flush=True)
            print(f"[DEBUG val] label sample : {batch_data_dict['label'][0]}", flush=True)
            print(f"[DEBUG val] label_vector shape : {target_dict['label_vector'].shape}", flush=True)
            print(f"[DEBUG val] label_vector sample:\n{target_dict['label_vector'][0]}", flush=True)
    
            # silence / fake 统计
            is_silence = (target_dict['label_vector'] == 0).all(dim=2)   # [B, S]
            target_power = target_dict['waveform'].float().pow(2).flatten(start_dim=2).mean(dim=2)  # [B, S]
            is_fake = (target_power < 1e-10) & ~is_silence
            print(f"[DEBUG val] is_silence per source : {is_silence.float().mean(dim=0).tolist()}", flush=True)
            print(f"[DEBUG val] is_fake    per source : {is_fake.float().mean(dim=0).tolist()}", flush=True)
            print(f"[DEBUG val] target_power per source : {target_power.mean(dim=0).tolist()}", flush=True)
    
            # loss 原始值
            loss_dict_raw = self.loss_func(output_dict, target_dict)
            print(f"[DEBUG val] raw loss : {loss_dict_raw['loss'].item():.6f}", flush=True)
            print(f"{'='*60}\n", flush=True)
        # ── END DEBUG ──────────────────────────────────────────────────
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict
