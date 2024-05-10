################## Github[loss.py]: https://github.com/ming024/FastSpeech2/blob/master/model/loss.py ###################
import torch
import torch.nn as nn


#################################### @ train.py ##############################################
class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        self.use_postnet = model_config["fastspeech_two"]["use_posetnet"]
        

    def forward(self, inputs, predictions):
        ### inputs: DataLoader's Batch
        ### inputs = [ids, raw_texts, speakers, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, pitches, energies, durations ]
        ### inputs[6:]: mels, mel_lens, max_mel_len, pitches, energies, durations 
        (mel_targets, _, _, pitch_targets, energy_targets, duration_targets, ) = inputs[6:]

        ### predictions: outputs from fastspeech2
        ### output_fs2, postnet_output_fs2, p_preds_fs2, e_preds_fs2, log_d_preds_fs2, d_rounded_fs2, src_masks_fs2, mel_masks_fs2, src_lens_fs2, mel_lens_fs2 
        (mel_predictions, postnet_mel_predictions, pitch_predictions, energy_predictions, log_duration_predictions, _, src_masks, mel_masks, _, _, ) = predictions
        
        ## minus_lize: True -> False, False -> True
        src_masks = ~src_masks
        mel_masks = ~mel_masks

        ## duration_targets: [16. 194]
        ## log_duration_targets: [16, 194]
        log_duration_targets = torch.log(duration_targets.float() + 1)

        ## mel_masks: [16, 1000] -> [16, 1000]
        ## mel_targets: [16, 1526, 80] -> [16, 1000, 80]
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]


        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        # src_masks: [16, 194]
        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
            ## pitch_predictions: [16, 194] -> [2699]
            ## pitch_targets: [16, 194] -> [2699]
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions[:, :mel_masks.shape[1]] ## Added for matching the shape
            pitch_predictions = pitch_predictions.masked_select(mel_masks)

            pitch_targets = pitch_targets[:, :mel_masks.shape[1]] ## Added for matching the shape
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
            ## energy_predictions: [16, 194] -> [2699]
            ## energy_targets: [16, 194] -> [2699]
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions[:, :mel_masks.shape[1]] ## Added for matching the shape
            energy_predictions = energy_predictions.masked_select(mel_masks)
            
            energy_targets = energy_targets[:, :mel_masks.shape[1]] ## Added for matching the shape
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        ## mel_predictions: [16, 1000, 80] -> [1280000]
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))

        ## mel_targets: [16, 1000, 80] -> [1280000]
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        # mel_loss: (tensor(6.0874, grad_fn=<MeanBackward0>),)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        ## pitch_loss: tensor(4.2568, grad_fn=<MseLossBackward0>),
        ## energy_loss: tensor(1.8249, grad_fn=<MseLossBackward0>),
        ## duration_loss: tensor(5.1772, grad_fn=<MseLossBackward0>))

        if self.use_postnet:

            ## postnet_mel_predictions: [16, 1000, 80] -> [1280000]
            postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))

            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
            # postnet_mel_loss: (tensor(6.0874, grad_fn=<MeanBackward0>),)

            total_loss = (mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss)

            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                duration_loss,
            )

        else:

            ## without postnet
            total_loss = (mel_loss + duration_loss + pitch_loss + energy_loss)
        
            return (
                total_loss,
                mel_loss,
                mel_loss,
                pitch_loss,
                energy_loss,
                duration_loss,)
