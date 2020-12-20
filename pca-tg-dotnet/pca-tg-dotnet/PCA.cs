using System;
using System.Collections.Generic;
using System.IO;
using Emgu.CV;

namespace pca_tg_dotnet
{
    internal class PCA
    {
        public PCA()
        {
        }

        internal void Run()
        {
            Console.WriteLine("Começou!");

            var p = 7;
            var train = new List<Person>();
            var test = new List<Person>();

            this.LoadDataset(@"..\..\..\dataset\ORL", train, test, p);

            Console.WriteLine("Treino: {0} / Teste: {1}", train.Count, test.Count);

			List<Person> otherOnes = this.LoadDatasetFromDir(@"..\..\dataset\Outros");
			test.AddRange(otherOnes);

			int start = 2;
			int end = 10;

			const int MAX_REC = 3500;
			const int MAX_DIS = 1700;

			for (int k = start; k <= end; k++)
			{
				PCAEigenFace model = new PCAEigenFace(k);
				model.Train(train);

				double minDis = Double.MaxValue;
				double maxDis = Double.MinValue;
				double minRec = Double.MaxValue;
				double maxRec = Double.MinValue;

				int trueNegativesCount = 0;
				int truePositivesCount = 0;

				int corrects = 0;
				int corrects2 = 0;
				foreach (Person personTest in test)
				{
					Mat testData = personTest.Data;
					int[] label = new int[1];
					double[] confidence = new double[1];
					double[] reconstructionError = new double[1];

					model.Predict(testData, label, confidence, reconstructionError);

					bool labelOk = label[0] == personTest.Label;
					if (labelOk) corrects++;

					if (reconstructionError[0] > MAX_REC)
					{
						Console.Error.WriteLine("NOT A PERSON - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
						if (!labelOk) trueNegativesCount++;
					}
					else if (confidence[0] > MAX_DIS)
					{
						Console.Error.WriteLine("UNKNOWN PEOPLE (by distance) - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
						if (!labelOk) trueNegativesCount++;
					}
					else if (reconstructionError[0] > 2400 && confidence[0] > 1500)
					{
						Console.Error.WriteLine("UNKNOWN PEOPLE (by two factors) - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
						if (!labelOk) trueNegativesCount++;
					}
					else if (labelOk)
					{
						truePositivesCount++;
					}
					else
					{
						Console.Error.WriteLine("UNKNOWN - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
					}

					if (labelOk && personTest.Id <= 400)
					{
						if (confidence[0] < minDis)
						{
							minDis = confidence[0];
						}
						if (confidence[0] > maxDis)
						{
							maxDis = confidence[0];
						}
						if (reconstructionError[0] < minRec)
						{
							minRec = reconstructionError[0];
						}
						if (reconstructionError[0] > maxRec)
						{
							maxRec = reconstructionError[0];
						}
					}

					if (personTest.Label == 182)
					{
						Console.WriteLine("Label: {0}, confidence: {1}, reconstructionError:{2}",
							label[0], confidence[0], reconstructionError[0]);
					}
				}

				Console.WriteLine("{0} correct: {1} incorrect", corrects, corrects2);

				int trues = truePositivesCount + trueNegativesCount;
				double accuracy = (double)trues / test.Count * 100;
				Console.WriteLine("K={0}, taxa de acerto={1}", k, accuracy);

				//double x = corrects / (double) test.size() * 100;
				Console.WriteLine("minDis={0}, maxDis={1}, minRec={2}, maxRec={3}",
					minDis, maxDis, minRec, maxRec);
			}

			Console.WriteLine("Terminou!");
        }

        private void eigenFaceRecognizerTest()
        {
            throw new NotImplementedException();
        }

        private List<Person> LoadDatasetFromDir(string path)
        {
			var data = new List<Person>();

			foreach (var file in Directory.GetFiles(path, "*.jpg"))
			{
				var person = Person.fromFile(file);

				data.Add(person);
			}

			return data;
		}

		private void LoadDataset(string path, List<Person> train, List<Person> test, int p)
        {
            var data = this.LoadDatasetFromDir(path);

            data.Sort();

            Random ran = new Random();
            int numSamplesPerPerson = 10;

            List<Person> samples = new List<Person>(numSamplesPerPerson);
            data.ForEach(person => {
                samples.Add(person);
                if (samples.Count == numSamplesPerPerson)
                {
                    while (samples.Count > p)
                    {
                        int index = ran.Next(samples.Count);
                        test.Add(samples[index]);
                        samples.RemoveAt(index);
                    }

                    if (p == numSamplesPerPerson)
                    {
                        test.AddRange(samples);
                    }

                    train.AddRange(samples);
                    samples.Clear();
                }
            });
        }
    }
}