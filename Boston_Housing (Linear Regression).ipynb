{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "BostonHousing_LinR.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "iLbZZIhPjEJV"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import sklearn\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.datasets import load_boston\n",
        "from sklearn.metrics import mean_squared_error,r2_score"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WJb8Soyfjz-b"
      },
      "source": [
        "boston = load_boston()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5u9-OQbnj3FR",
        "outputId": "e76c7463-1f88-4877-a76c-f4a38c7b1191"
      },
      "source": [
        "print(boston)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "{'data': array([[6.3200e-03, 1.8000e+01, 2.3100e+00, ..., 1.5300e+01, 3.9690e+02,\n",
            "        4.9800e+00],\n",
            "       [2.7310e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9690e+02,\n",
            "        9.1400e+00],\n",
            "       [2.7290e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9283e+02,\n",
            "        4.0300e+00],\n",
            "       ...,\n",
            "       [6.0760e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,\n",
            "        5.6400e+00],\n",
            "       [1.0959e-01, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9345e+02,\n",
            "        6.4800e+00],\n",
            "       [4.7410e-02, 0.0000e+00, 1.1930e+01, ..., 2.1000e+01, 3.9690e+02,\n",
            "        7.8800e+00]]), 'target': array([24. , 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9, 15. ,\n",
            "       18.9, 21.7, 20.4, 18.2, 19.9, 23.1, 17.5, 20.2, 18.2, 13.6, 19.6,\n",
            "       15.2, 14.5, 15.6, 13.9, 16.6, 14.8, 18.4, 21. , 12.7, 14.5, 13.2,\n",
            "       13.1, 13.5, 18.9, 20. , 21. , 24.7, 30.8, 34.9, 26.6, 25.3, 24.7,\n",
            "       21.2, 19.3, 20. , 16.6, 14.4, 19.4, 19.7, 20.5, 25. , 23.4, 18.9,\n",
            "       35.4, 24.7, 31.6, 23.3, 19.6, 18.7, 16. , 22.2, 25. , 33. , 23.5,\n",
            "       19.4, 22. , 17.4, 20.9, 24.2, 21.7, 22.8, 23.4, 24.1, 21.4, 20. ,\n",
            "       20.8, 21.2, 20.3, 28. , 23.9, 24.8, 22.9, 23.9, 26.6, 22.5, 22.2,\n",
            "       23.6, 28.7, 22.6, 22. , 22.9, 25. , 20.6, 28.4, 21.4, 38.7, 43.8,\n",
            "       33.2, 27.5, 26.5, 18.6, 19.3, 20.1, 19.5, 19.5, 20.4, 19.8, 19.4,\n",
            "       21.7, 22.8, 18.8, 18.7, 18.5, 18.3, 21.2, 19.2, 20.4, 19.3, 22. ,\n",
            "       20.3, 20.5, 17.3, 18.8, 21.4, 15.7, 16.2, 18. , 14.3, 19.2, 19.6,\n",
            "       23. , 18.4, 15.6, 18.1, 17.4, 17.1, 13.3, 17.8, 14. , 14.4, 13.4,\n",
            "       15.6, 11.8, 13.8, 15.6, 14.6, 17.8, 15.4, 21.5, 19.6, 15.3, 19.4,\n",
            "       17. , 15.6, 13.1, 41.3, 24.3, 23.3, 27. , 50. , 50. , 50. , 22.7,\n",
            "       25. , 50. , 23.8, 23.8, 22.3, 17.4, 19.1, 23.1, 23.6, 22.6, 29.4,\n",
            "       23.2, 24.6, 29.9, 37.2, 39.8, 36.2, 37.9, 32.5, 26.4, 29.6, 50. ,\n",
            "       32. , 29.8, 34.9, 37. , 30.5, 36.4, 31.1, 29.1, 50. , 33.3, 30.3,\n",
            "       34.6, 34.9, 32.9, 24.1, 42.3, 48.5, 50. , 22.6, 24.4, 22.5, 24.4,\n",
            "       20. , 21.7, 19.3, 22.4, 28.1, 23.7, 25. , 23.3, 28.7, 21.5, 23. ,\n",
            "       26.7, 21.7, 27.5, 30.1, 44.8, 50. , 37.6, 31.6, 46.7, 31.5, 24.3,\n",
            "       31.7, 41.7, 48.3, 29. , 24. , 25.1, 31.5, 23.7, 23.3, 22. , 20.1,\n",
            "       22.2, 23.7, 17.6, 18.5, 24.3, 20.5, 24.5, 26.2, 24.4, 24.8, 29.6,\n",
            "       42.8, 21.9, 20.9, 44. , 50. , 36. , 30.1, 33.8, 43.1, 48.8, 31. ,\n",
            "       36.5, 22.8, 30.7, 50. , 43.5, 20.7, 21.1, 25.2, 24.4, 35.2, 32.4,\n",
            "       32. , 33.2, 33.1, 29.1, 35.1, 45.4, 35.4, 46. , 50. , 32.2, 22. ,\n",
            "       20.1, 23.2, 22.3, 24.8, 28.5, 37.3, 27.9, 23.9, 21.7, 28.6, 27.1,\n",
            "       20.3, 22.5, 29. , 24.8, 22. , 26.4, 33.1, 36.1, 28.4, 33.4, 28.2,\n",
            "       22.8, 20.3, 16.1, 22.1, 19.4, 21.6, 23.8, 16.2, 17.8, 19.8, 23.1,\n",
            "       21. , 23.8, 23.1, 20.4, 18.5, 25. , 24.6, 23. , 22.2, 19.3, 22.6,\n",
            "       19.8, 17.1, 19.4, 22.2, 20.7, 21.1, 19.5, 18.5, 20.6, 19. , 18.7,\n",
            "       32.7, 16.5, 23.9, 31.2, 17.5, 17.2, 23.1, 24.5, 26.6, 22.9, 24.1,\n",
            "       18.6, 30.1, 18.2, 20.6, 17.8, 21.7, 22.7, 22.6, 25. , 19.9, 20.8,\n",
            "       16.8, 21.9, 27.5, 21.9, 23.1, 50. , 50. , 50. , 50. , 50. , 13.8,\n",
            "       13.8, 15. , 13.9, 13.3, 13.1, 10.2, 10.4, 10.9, 11.3, 12.3,  8.8,\n",
            "        7.2, 10.5,  7.4, 10.2, 11.5, 15.1, 23.2,  9.7, 13.8, 12.7, 13.1,\n",
            "       12.5,  8.5,  5. ,  6.3,  5.6,  7.2, 12.1,  8.3,  8.5,  5. , 11.9,\n",
            "       27.9, 17.2, 27.5, 15. , 17.2, 17.9, 16.3,  7. ,  7.2,  7.5, 10.4,\n",
            "        8.8,  8.4, 16.7, 14.2, 20.8, 13.4, 11.7,  8.3, 10.2, 10.9, 11. ,\n",
            "        9.5, 14.5, 14.1, 16.1, 14.3, 11.7, 13.4,  9.6,  8.7,  8.4, 12.8,\n",
            "       10.5, 17.1, 18.4, 15.4, 10.8, 11.8, 14.9, 12.6, 14.1, 13. , 13.4,\n",
            "       15.2, 16.1, 17.8, 14.9, 14.1, 12.7, 13.5, 14.9, 20. , 16.4, 17.7,\n",
            "       19.5, 20.2, 21.4, 19.9, 19. , 19.1, 19.1, 20.1, 19.9, 19.6, 23.2,\n",
            "       29.8, 13.8, 13.3, 16.7, 12. , 14.6, 21.4, 23. , 23.7, 25. , 21.8,\n",
            "       20.6, 21.2, 19.1, 20.6, 15.2,  7. ,  8.1, 13.6, 20.1, 21.8, 24.5,\n",
            "       23.1, 19.7, 18.3, 21.2, 17.5, 16.8, 22.4, 20.6, 23.9, 22. , 11.9]), 'feature_names': array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',\n",
            "       'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7'), 'DESCR': \".. _boston_dataset:\\n\\nBoston house prices dataset\\n---------------------------\\n\\n**Data Set Characteristics:**  \\n\\n    :Number of Instances: 506 \\n\\n    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\\n\\n    :Attribute Information (in order):\\n        - CRIM     per capita crime rate by town\\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\\n        - INDUS    proportion of non-retail business acres per town\\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\\n        - NOX      nitric oxides concentration (parts per 10 million)\\n        - RM       average number of rooms per dwelling\\n        - AGE      proportion of owner-occupied units built prior to 1940\\n        - DIS      weighted distances to five Boston employment centres\\n        - RAD      index of accessibility to radial highways\\n        - TAX      full-value property-tax rate per $10,000\\n        - PTRATIO  pupil-teacher ratio by town\\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\\n        - LSTAT    % lower status of the population\\n        - MEDV     Median value of owner-occupied homes in $1000's\\n\\n    :Missing Attribute Values: None\\n\\n    :Creator: Harrison, D. and Rubinfeld, D.L.\\n\\nThis is a copy of UCI ML housing dataset.\\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/housing/\\n\\n\\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\\n\\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\\nprices and the demand for clean air', J. Environ. Economics & Management,\\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\\npages 244-261 of the latter.\\n\\nThe Boston house-price data has been used in many machine learning papers that address regression\\nproblems.   \\n     \\n.. topic:: References\\n\\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\\n\", 'filename': '/usr/local/lib/python3.7/dist-packages/sklearn/datasets/data/boston_house_prices.csv'}\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fsvwApNkkFn9",
        "outputId": "6c36943f-c73b-4104-ba69-03d9391bad8e"
      },
      "source": [
        "boston.keys()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rXOcGufKkL2U",
        "outputId": "24e67548-a32a-4f74-8450-0e4076784726"
      },
      "source": [
        "print(boston.data.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(506, 13)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Sa8gq6vokaJh",
        "outputId": "d703a0f0-5004-4e63-8ba1-798f1388e9c0"
      },
      "source": [
        "bos=pd.DataFrame(boston.data)\n",
        "print(bos.head(2))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "        0     1     2    3      4   ...   8      9     10     11    12\n",
            "0  0.00632  18.0  2.31  0.0  0.538  ...  1.0  296.0  15.3  396.9  4.98\n",
            "1  0.02731   0.0  7.07  0.0  0.469  ...  2.0  242.0  17.8  396.9  9.14\n",
            "\n",
            "[2 rows x 13 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kGGuhl9Nkrh4",
        "outputId": "606c1981-8926-49ac-d3bc-06df23e214cf"
      },
      "source": [
        "print(boston.feature_names)\n",
        "bos.columns = boston.feature_names"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'\n",
            " 'B' 'LSTAT']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kI719jrik-ar",
        "outputId": "a1a0141a-886c-4523-8df4-1b9329e1952a"
      },
      "source": [
        "bos['PRICE']=boston.target\n",
        "print(bos.shape)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(506, 14)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FoB2hKSglMeY",
        "outputId": "853117d3-0165-449d-b881-0f60b5182ac7"
      },
      "source": [
        "Y = bos['PRICE']\n",
        "X = bos.drop('PRICE',axis=1)\n",
        "print(Y.head(5))\n",
        "print(X.head(5))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0    24.0\n",
            "1    21.6\n",
            "2    34.7\n",
            "3    33.4\n",
            "4    36.2\n",
            "Name: PRICE, dtype: float64\n",
            "      CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT\n",
            "0  0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98\n",
            "1  0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14\n",
            "2  0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03\n",
            "3  0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94\n",
            "4  0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33\n",
            "\n",
            "[5 rows x 13 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ic8pKdkzmF3H"
      },
      "source": [
        "X_train,X_test,Y_train,Y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.3,random_state=5)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y448rXYFmhax"
      },
      "source": [
        "model = LinearRegression()\n",
        "model.fit(X_train,Y_train)\n",
        "Y_train_pred = model.predict(X_train)\n",
        "Y_test_pred = model.predict(X_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 233
        },
        "id": "clvnRH2BnI-O",
        "outputId": "267fc457-459d-4fe0-ab61-2d3f12c8354f"
      },
      "source": [
        "df = pd.DataFrame(Y_test_pred,Y_test)\n",
        "df.head(5)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>PRICE</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>37.6</th>\n",
              "      <td>37.389977</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27.9</th>\n",
              "      <td>31.567942</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>22.6</th>\n",
              "      <td>27.133739</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13.8</th>\n",
              "      <td>6.551176</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>35.2</th>\n",
              "      <td>33.693108</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "               0\n",
              "PRICE           \n",
              "37.6   37.389977\n",
              "27.9   31.567942\n",
              "22.6   27.133739\n",
              "13.8    6.551176\n",
              "35.2   33.693108"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UL2eYqvinW_z",
        "outputId": "3674f748-cf71-4ac9-8ec3-d432f1652b9a"
      },
      "source": [
        "mse = mean_squared_error(Y_test,Y_test_pred)\n",
        "print(mse)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "30.697037704088636\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "uPdaURmCn57h",
        "outputId": "f794a575-10ab-4c89-ed63-b485b2d1af69"
      },
      "source": [
        "plt.scatter(Y_train,Y_train_pred,c='blue',marker='o',label='Training Data')\n",
        "plt.scatter(Y_test,Y_test_pred,c='red',marker='+',label='Test Data')\n",
        "plt.xlabel('True values')\n",
        "plt.ylabel('Predicted values')\n",
        "plt.legend(loc='upper left')\n",
        "plt.plot()\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9nho8BBapRjF"
      },
      "source": [
        "import pickle"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rT9ElrgUpTrQ"
      },
      "source": [
        "with open('model_pickle','wb') as f:\n",
        "  pickle.dump(model,f)"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
