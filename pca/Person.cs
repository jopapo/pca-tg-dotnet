using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Drawing;
using System.IO;

namespace Furb.Pos.DataScience.PCA
{
    internal class Person : IComparable<Person>
    {
        public int Id { private set; get; }
        public int Label { private set; get; }
        public Mat Data { private set; get; }

        internal static Person fromFile(string file)
        {
			var filename = Path.GetFileNameWithoutExtension(file);
            var fileparts = filename.Split("_");

            var person = new Person()
            {
                Id = Convert.ToInt32(fileparts[0]),
                Label = Convert.ToInt32(fileparts[1])
            };

			var image_data = CvInvoke.Imread(file, ImreadModes.Grayscale);

			var dst = new Mat();

			var size = new Size(80, 80);

			CvInvoke.Resize(image_data, dst, size, 0, 0, Inter.Linear);

			// De imagem com 8 bits, sem sinal, 1 canal
            dst = dst.T().Reshape(1, dst.Cols * dst.Rows);

            person.Data = new Mat();

			// Para imagem com 64 bits, com sinal e ponto flutuante, 1 canal
			dst.ConvertTo(person.Data, DepthType.Cv64F, 1, 0);

            return person;

		}

        public int CompareTo(Person other)
        {
            return this.Id - other.Id;
        }
    }
}