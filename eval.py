import numpy as np
import yaml
from src.utils.torch_ioueval import iouEval


def evalutation(preds, labels, DATA):
    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    # make lookup table for mapping
    maxkey = max(class_remap.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())

    ignore = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16, 18, 19]
    print("Ignoring xentropy class ", ignore, " in IoU evaluation")

    evaluator = iouEval(nr_classes, ignore)
    evaluator.reset()

    progress = 10
    count = 0
    print("Evaluating sequences: ", end="", flush=True)
    # open each file, get the tensor, and make the iou comparison
    for label, pred in zip(labels, preds):
        count += 1
        # pred ground label to original label
        ground_label = [9, 11, 12, 17]
        tmp = np.zeros_like(pred)
        tmp[np.isin(label, ground_label)] = 1
        pred = np.where(np.logical_and(pred == 1, tmp), label, 0)
        evaluator.addBatch(pred.astype(np.int64), label.astype(np.int64))

    # when I am done, print the evaluation
    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()
    m_recall = evaluator.getrecall()

    print(
        "Validation set:\n"
        "Acc avg {m_accuracy:.3f}\n"
        "IoU avg {m_jaccard:.3f}\n"
        "Recall avg {m_recall:.3f}".format(m_accuracy=m_accuracy, m_jaccard=m_jaccard, m_recall=m_recall)
    )
    # print also classwise
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            print(
                "IoU class {i:} [{class_str:}] = {jacc:.3f}".format(
                    i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc
                )
            )
    return m_accuracy, m_jaccard, m_recall
