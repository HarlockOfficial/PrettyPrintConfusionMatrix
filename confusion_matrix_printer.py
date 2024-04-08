import numpy as np
import pandas as pd

from pretty_confusion_matrix import pp_matrix

arr = np.array([])
idx = ['Feet', 'Left Hand', 'Right Hand']
df = pd.DataFrame(arr, index=idx, columns=idx)
plt, ax = pp_matrix(df, pred_val_axis='x', show_null_values=2, annot=True, fz=10, lw=0.5, cbar=True, cmap='Blues',
                    title='Confusion Matrix MDM')
plt.savefig('./figures/confusion_matrix_mdm.png')
