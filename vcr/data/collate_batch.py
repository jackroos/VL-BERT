import torch

from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.task = self.dataset.task
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
        max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        max_masks = max([data[self.data_names.index('masks')].shape[0] for data in batch])
        if self.test_mode and self.task == 'QA2R':
            max_question_length = max([len(q) for data in batch for q in data[self.data_names.index('question')]])
        else:
            max_question_length = max([len(data[self.data_names.index('question')]) for data in batch])
        if 'answer_choices' in self.data_names:
            max_answer_length = max([len(answer) for data in batch for answer in data[self.data_names.index('answer_choices')]])
        if 'answer' in self.data_names:
            max_answer_length = max([len(data[self.data_names.index('answer')]) for data in batch])
        if 'rationale_choices' in self.data_names:
            max_rationale_length = max([len(rationale) for data in batch for rationale in data[self.data_names.index('rationale_choices')]])
        if 'rationale' in self.data_names:
            max_rationale_length = max([len(data[self.data_names.index('rationale')]) for data in batch])
        if 'question_align_matrix' in self.data_names:
            if self.test_mode and self.task == 'QA2R':
                max_q_align_length = max([m.shape[0]
                                          for data in batch
                                          for m in data[self.data_names.index('question_align_matrix')]])
            else:
                max_q_align_length = max([data[self.data_names.index('question_align_matrix')].shape[0] for data in batch])
        if 'answer_align_matrix' in self.data_names:
            if isinstance(batch[0][self.data_names.index('answer_align_matrix')], list) or \
                    batch[0][self.data_names.index('answer_align_matrix')].dim() == 3:
                max_a_align_length = max([m.shape[0]
                                          for data in batch
                                          for m in data[self.data_names.index('answer_align_matrix')]])
            elif batch[0][self.data_names.index('answer_align_matrix')].dim() == 2:
                max_a_align_length = max([data[self.data_names.index('answer_align_matrix')].shape[0]
                                          for data in batch])
            else:
                raise ValueError("invalid dims of answer_align_matrix")
        if 'rationale_align_matrix' in self.data_names:
            if isinstance(batch[0][self.data_names.index('rationale_align_matrix')], list) or \
                    batch[0][self.data_names.index('rationale_align_matrix')].dim() == 3:
                max_r_align_length = max([m.shape[0]
                                          for data in batch
                                          for m in data[self.data_names.index('rationale_align_matrix')]])
            elif batch[0][self.data_names.index('rationale_align_matrix')].dim() == 2:
                max_r_align_length = max([data[self.data_names.index('rationale_align_matrix')].shape[0]
                                          for data in batch])
            else:
                raise ValueError("invalid dims of rationale_align_matrix!")

        for i, ibatch in enumerate(batch):
            out = {}
            image = ibatch[self.data_names.index('image')]
            out['image'] = clip_pad_images(image, max_shape, pad=0)

            boxes = ibatch[self.data_names.index('boxes')]
            out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-1)

            masks = ibatch[self.data_names.index('masks')]
            mask_height, mask_width = masks.shape[1:]
            out['masks'] = clip_pad_boxes(masks.view(masks.shape[0], -1), max_masks, pad=-1).view(-1, mask_height, mask_width)

            question = ibatch[self.data_names.index('question')]
            if self.test_mode and self.task == 'QA2R':
                out['question'] = torch.stack(tuple(clip_pad_2d(q, (max_question_length, len(q[0])), pad=-2) for q in question),
                                              dim=0)
                if 'question_align_matrix' in self.data_names:
                    q_align_matrix = ibatch[self.data_names.index('question_align_matrix')]
                    out['question_align_matrix'] = torch.stack(
                        tuple(clip_pad_2d(m, (max_q_align_length, max_question_length), pad=0) for m in q_align_matrix),
                        dim=0)
            else:
                out['question'] = clip_pad_2d(question, (max_question_length, len(question[0])), pad=-2)
                if 'question_align_matrix' in self.data_names:
                    q_align_matrix = ibatch[self.data_names.index('question_align_matrix')]
                    out['question_align_matrix'] = clip_pad_2d(q_align_matrix,
                                                               (max_q_align_length, max_question_length),
                                                               pad=0)
            if 'answer' in self.data_names:
                answer = ibatch[self.data_names.index('answer')]
                out['answer'] = clip_pad_2d(answer, (max_answer_length, len(answer[0])), pad=-2)
            if 'answer_choices' in self.data_names:
                answer_choices = ibatch[self.data_names.index('answer_choices')]
                out['answer_choices'] = torch.stack(tuple(clip_pad_2d(answer,
                                                                      (max_answer_length, len(answer[0])),
                                                                      pad=-2)
                                                    for answer in answer_choices),
                                                    dim=0)
            if 'answer_align_matrix' in self.data_names:
                a_align_matrix = ibatch[self.data_names.index('answer_align_matrix')]
                if isinstance(a_align_matrix, list) or a_align_matrix.dim() == 3:
                    out['answer_align_matrix'] = torch.stack(
                        tuple(clip_pad_2d(m, (max_a_align_length, max_answer_length), pad=0) for m in a_align_matrix),
                        dim=0)
                elif a_align_matrix.dim() == 2:
                    out['answer_align_matrix'] = clip_pad_2d(a_align_matrix, (max_a_align_length, max_answer_length),
                                                             pad=0)
            if 'rationale' in self.data_names:
                rationale = ibatch[self.data_names.index('rationale')]
                out['rationale'] = clip_pad_2d(rationale, (max_rationale_length, len(rationale[0])), pad=-2)
            if 'rationale_choices' in self.data_names:
                rationale_choices = ibatch[self.data_names.index('rationale_choices')]
                out['rationale_choices'] = torch.stack(tuple(clip_pad_2d(rationale,
                                                                         (max_rationale_length, len(rationale[0])),
                                                                         pad=-2)
                                                       for rationale in rationale_choices),
                                                       dim=0)
            if 'rationale_align_matrix' in self.data_names:
                r_align_matrix = ibatch[self.data_names.index('rationale_align_matrix')]
                if isinstance(r_align_matrix, list) or r_align_matrix.dim() == 3:
                    out['rationale_align_matrix'] = torch.stack(
                        tuple(clip_pad_2d(m, (max_r_align_length, max_rationale_length), pad=0) for m in r_align_matrix),
                        dim=0)
                elif r_align_matrix.dim() == 2:
                    out['rationale_align_matrix'] = clip_pad_2d(r_align_matrix,
                                                                (max_r_align_length, max_rationale_length), pad=0)

            if 'answer_label' in self.data_names:
                out['answer_label'] = ibatch[self.data_names.index('answer_label')]
            if 'rationale_label' in self.data_names:
                out['rationale_label'] = ibatch[self.data_names.index('rationale_label')]
            out['im_info'] = ibatch[self.data_names.index('im_info')]

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if isinstance(items[0], torch.Tensor):
                out_tuple += (torch.stack(tuple(items), dim=0), )
            else:
                out_tuple += (list(items), )

        return out_tuple

