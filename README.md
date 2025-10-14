# Candlestick Matcher Backend

Receives the user uploaded chart and compares it with known candlestick patterns, returning the one with the highest sim score.

## How it works

I used ResNet18 feature extractor to precompute the vector embeddings of top candlestick patterns. Using a sliding window, I extracted embeddings from patches of the uploaded chart and used cosine sim to find the higest score between a patch and a pattern.

Using OpenCV, I can find the number of candlesticks in the chart and the height of the longest candlestick. Along with the number of candlesticks in each pattern, these variables are then used to determine the size and stride of the sliding window to perform more meaningful comparisons.

### Limitations

Without knowing the height of the pattern in the chart, I used the longest height to account for all patterns. This however led to inaccurate matchings. Tests where I knew the height of the pattern and specified it in the dimensions of the window produced better results.

To achieve acceptable computing speeds (due to limited free resources), I increased the window stride. This means less overlap between patches and correspoindingly less truthful results since the highest sim score might not be reached.

I used a pretrained model for this project. Using a model tuned for my specific use case will produce better results but this is just proof of concept project. I also do not have access to a reasonable sized and accurate dataset to perform the training and fine-tuning myself.

## Moved

Render did could not handle OpenCV and ResNet and crashes everytime I call the api, so I moved to Hugging Face and it finally worked.

You can find the repo [here](https://huggingface.co/spaces/jaey8den/candlestick-matcher-backend).
