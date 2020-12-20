using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.IO;

namespace Furb.Pos.DataScience.PCA
{
    internal class PCAEigenFace
    {
        private int numComponents;
        private Mat mean; // Vai produção
        private Mat diffs;
        private Mat covariance;
        private Mat eigenvalues;
        private Mat eigenvectors;
        private Mat eigenfaces; // Vai produção
        private Mat projections; // Vai produção
        private int[] labels; // Vai produção

        private readonly bool DEBUG = true;

        public PCAEigenFace(int numComponents)
        {
            this.numComponents = numComponents;
        }

        internal void Train(List<Person> train)
        {
            CalcMean(train);
            CalcDiff(train);
            CalcCovariance();
            CalcEigen();
            CalcEigenFaces();
            CalcProjections(train);
        }

        private void CalcProjections(List<Person> train)
        {
            labels = new int[train.Count]; // 1 label para cada imagem de treino.
            projections = new Mat(this.numComponents, train.Count, DepthType.Cv64F, 1);
            for (int j = 0; j < diffs.Cols; j++)
            {
                Mat diff = diffs.Col(j);
                Mat w/*k x 1*/ = this.Mul(eigenfaces.T(), diff); // U=(6400 x k)t, Ut=k x 6400 * diff= 6400 x 1 = w=k x 1
                w.CopyTo(projections.Col(j));
                labels[j] = train[j].Label;
            }
        }

        private Mat Mul(Mat a, Mat b)
        {
            Mat c = new Mat(a.Rows, b.Cols, DepthType.Cv64F, 1);
            CvInvoke.Gemm(a, b, 1, new Mat(), 1, c);
            return c;
        }

        private void CalcEigenFaces()
        {
            Mat evt = eigenvectors.T();
            int k = numComponents > 0 ? numComponents : evt.Cols;
            numComponents = k;
            Mat ev_k = new Mat(evt.Rows, k, DepthType.Cv64F, 1);
            // Mat ev_k = evt.ColRange(0, k);
            for (int j = 0; j < ev_k.Cols; j++)
            {
                evt.Col(j).CopyTo(ev_k.Col(j));
            }

            eigenfaces = this.Mul(diffs, ev_k);
            // k =3
            // 1 9 7
            // 4 5 8
            // 7 8 0
            // 6 1 7
            for (int j = 0; j < eigenfaces.Cols; j++)
            {
                Mat ef = eigenfaces.Col(j);
                // Normalização L2 = Yi = Xi / sqrt(sum(Xi)^2)), onde i = 0, ... rows-1
                CvInvoke.Normalize(ef, ef);
            }

            PrintEigenFaces();
        }

        private void PrintEigenFaces()
        {
            for (int j = 0; j < eigenfaces.Cols; j++)
            {
                Mat y = new Mat(eigenfaces.Rows, 1, eigenfaces.Depth, eigenfaces.NumberOfChannels);
                eigenfaces.Col(j).CopyTo(y.Col(0));
                if (DEBUG) {
                    this.SaveImage(y, "out" + (j + 1) + ".jpg");
                }
            }
        }

        private void CalcEigen()
        {
            eigenvalues = new Mat();
            eigenvectors = new Mat();
            CvInvoke.Eigen(covariance, eigenvalues, eigenvectors);
            PrintEigenValues();
        }

        private void PrintEigenValues()
        {
            // Soma os eigenvalues
            double sum = 0;
            for (int i = 0; i < eigenvalues.Rows; i++)
            {
                //sum += eigenvalues.get(i, 0)[0];
                sum += eigenvalues.GetDoubleValue(i, 0);
            }

            // Calcula o percentual de contribuição de cada eigenvalue na explicação dos dados.
            double acumulado = 0;
            for (int i = 0; i < eigenvalues.Rows; i++)
            {
                //double v = eigenvalues.get(i, 0)[0];
                double v = eigenvalues.GetDoubleValue(i, 0);
                double percentual = v / sum * 100;
                acumulado += percentual;
                Console.WriteLine("CP {0}, percentual: {1} {2}", (i + 1), percentual, acumulado);
            }
        }

        private void CalcCovariance()
        {
            covariance = this.Mul(diffs.T(), diffs);
        }

        private void CalcDiff(List<Person> train)
        {
            Mat sample = train[0].Data;
            // 1 9 7
            // 4 5 8
            // 7 8 0
            // 6 1 7
            diffs = new Mat(sample.Rows, train.Count, sample.Depth, sample.NumberOfChannels);
            for (int i = 0; i < diffs.Rows; i++)
            {
                for (int j = 0; j < diffs.Cols; j++)
                {
                    //double mv = mean.get(i, 0)[0];
                    double mv = mean.GetDoubleValue(i, 0);
                    Mat data = train[j].Data;
                    //double pv = data.get(i, 0)[0];
                    double pv = data.GetDoubleValue(i, 0);
                    double v = pv - mv;
                    //diffs.put(i, j, v);
                    diffs.SetDoubleValue(i, j, v);
                }
            }
        }

        private void CalcMean(List<Person> train)
        {
            Mat sample = train[0].Data;
            mean = Mat.Zeros(/*6400*/sample.Rows, /*1*/sample.Cols, /*CvType.CV_64FC1*/sample.Depth, sample.NumberOfChannels);

            /// Begin Calculado na mão
            train.ForEach(person => {
                Mat data = person.Data;
                for (int i = 0; i < mean.Rows; i++)
                {
                    //double mv = mean.get(i, 0)[0]; // Obtém o valor da célula no primeiro canal.
                    //double pv = data.get(i, 0)[0]; // Obtém o valor da célula no primeiro canal.
                    double mv = mean.GetDoubleValue(i, 0); // Obtém o valor da célula no primeiro canal.
                    double pv = data.GetDoubleValue(i, 0); // Obtém o valor da célula no primeiro canal.
                    mv += pv;
                    //mean.put(i, 0, mv);
                    mean.SetDoubleValue(i, 0, mv);
                }
            });

            int M = train.Count;
            for (int i = 0; i < mean.Rows; i++)
            {
                //double mv = mean.get(i, 0)[0]; // Obtém o valor da célula no primeiro canal.
                double mv = mean.GetDoubleValue(i, 0); // Obtém o valor da célula no primeiro canal.
                mv /= M;
                //mean.put(i, 0, mv);
                mean.SetDoubleValue(i, 0, mv);
            }
            /// End Calculado na mão

            // Begin OpenCV
            // 1 9 7
            // 4 5 8
            // 7 8 0
            // 6 1 7
            //		Mat src = new Mat(sample.rows(), train.size(), sample.type());
            //		for (int i = 0; i < train.size(); i++) {
            //			train.get(i).getData().col(0).copyTo(src.col(i));
            //		}
            //		
            //		Mat mean2 = Mat.zeros(sample.rows(), sample.cols(), sample.type());
            //		Core.reduce(src, mean2, /*0=linha, 1=coluna*/1, Core.REDUCE_AVG, mean.type());
            // End OpenCV

            if (DEBUG) {
                SaveImage(mean, "mean1.jpg");
                //SaveImage(mean2, "D:\\PCA\\mean2.jpg");
            }
        }

        private void SaveImage(Mat image, String filename)
        {
            // [1,2,3,4,5]t
            // 1 2
            // 2 3
            Mat dst = new Mat();
            CvInvoke.Normalize(image, dst, 0, 255, NormType.MinMax, DepthType.Cv8U);

            // 6400 x 1
            // 80 x 80

            dst = dst.Reshape(1, 80);
            dst = dst.T();

            if (! Directory.Exists("out")) {
                Directory.CreateDirectory("out");
            }
            
            CvInvoke.Imwrite(Path.Combine("out", filename), dst);
        }

        internal void Predict(Mat testData, int[] label, double[] confidence, double[] reconstructionError)
        {
            Mat diff = new Mat();
            // Subtrai a imagem desconhecida, da imagem média.
            CvInvoke.Subtract(testData, mean, diff);

            // Projeta a imagem desconhecida, no mesmo espaço das images de treino.
            Mat w = this.Mul(eigenfaces.T(), diff);

            // Calcula a imagem de treino mais próxima da imagem desconhecida que foi projetada.
            int minJ = 0;
            double minDistance = CalcDistance(w, projections.Col(minJ));
            for (int j = 1; j < projections.Cols; j++)
            {
                double distance = CalcDistance(w, projections.Col(j));
                if (distance < minDistance)
                {
                    minDistance = distance;
                    minJ = j;
                }
            }

            // Obtém o label e a distância da imagem de treino mais próxima da imagem de teste
            // e as retorna como resposta
            label[0] = labels[minJ];
            confidence[0] = minDistance;

            // Calcular o erro de reconstrução.
            Mat reconstruction = CalcReconstruction(w);
            reconstructionError[0] = CvInvoke.Norm(testData, reconstruction, NormType.L2);
        }

        private Mat CalcReconstruction(Mat w)
        {
            Mat result = this.Mul(eigenfaces, w); //[eigenfaces=6400 x k] * [w=k x 1] = result=6400 x 1.
                                             // result += mean
            CvInvoke.Add(result, mean, result);
            return result;
        }

        private double CalcDistance(Mat p, Mat q)
        {
            // Calcula a distância euclidiana
            // d = sqrt(sum(pi - q1)^2))
            // 1
            // 2
            // 3
            double distance = 0;
            for (int i = 0; i < p.Rows; i++)
            {
                //double pi = p.get(i, 0)[0];
                // double qi = q.get(i, 0)[0];
                double pi = p.GetDoubleValue(i, 0);
                double qi = q.GetDoubleValue(i, 0);
                double d = pi - qi;
                distance += d * d;
            }

            //double d2 = Core.norm(p, q, Core.NORM_L2);
            distance = Math.Sqrt(distance);
            return distance;
        }
    }
}