using System;
using System.Collections.Generic;
using System.IO;
using Emgu.CV;
using Microsoft.Extensions.Logging;

namespace Furb.Pos.DataScience.PCA
{
    internal class PCABoostrap
    {

		private readonly ILogger logger;
		private readonly PCAEigenFace eigenFaceModel;
		
        public PCABoostrap(ILogger<PCABoostrap> logger, PCAEigenFace eigenFaceModel)
        {
			this.logger = logger;
			this.eigenFaceModel = eigenFaceModel;
		}

		internal void Run(int p)
        {
			logger.LogDebug("Começou!");

            var train = new List<Person>();
            var test = new List<Person>();

            this.LoadDataset(@"dataset\ORL", train, test, p);

			logger.LogDebug("Treino: {0} / Teste: {1}", train.Count, test.Count);

			List<Person> otherOnes = this.LoadDatasetFromDir(@"dataset\Outros");
			test.AddRange(otherOnes);

			int start = 10;
			int end = 20;

			const int MAX_REC = 3500;
			const int MAX_DIS = 1700;

			for (int k = start; k <= end; k++)
			{
				this.eigenFaceModel.Train(train, k);

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

					this.eigenFaceModel.Predict(testData, label, confidence, reconstructionError);

					bool labelOk = label[0] == personTest.Label;
					if (labelOk) corrects++;

					if (reconstructionError[0] > MAX_REC)
					{
						logger.LogDebug("NOT A PERSON - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
						if (!labelOk) trueNegativesCount++;
					}
					else if (confidence[0] > MAX_DIS)
					{
						logger.LogDebug("UNKNOWN PEOPLE (by distance) - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
						if (!labelOk) trueNegativesCount++;
					}
					else if (reconstructionError[0] > 2400 && confidence[0] > 1500)
					{
						logger.LogDebug("UNKNOWN PEOPLE (by two factors) - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
							label[0], confidence[0], reconstructionError[0], personTest.Label);
						if (!labelOk) trueNegativesCount++;
					}
					else if (labelOk)
					{
						truePositivesCount++;
					}
					else
					{
						logger.LogDebug("UNKNOWN - Predicted label {0}, confidence: {1}, reconstructionError: {2}, original label: {3}",
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
						logger.LogDebug("Label: {0}, confidence: {1}, reconstructionError:{2}",
							label[0], confidence[0], reconstructionError[0]);
					}
				}

				logger.LogDebug("{0} correct: {1} incorrect", corrects, corrects2);

				int trues = truePositivesCount + trueNegativesCount;
				double accuracy = (double)trues / test.Count * 100;
				logger.LogInformation("{0} componentes principais, acurácia: {1}", k, accuracy);

				logger.LogDebug("minDis={0}, maxDis={1}, minRec={2}, maxRec={3}",
					minDis, maxDis, minRec, maxRec);
			}

			logger.LogDebug("Terminou!");
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