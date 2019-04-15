# mkldnn_unit_test
## How to change the environment variable
1. For python backend:
```bash
import os
os.environ['MKLDNN'] = '0'
```
2. For C++ backend:
```bash
#include <stdlib.h>

if (*getenv("MKLDNN") == '0'){
  do some thing
}
```
