using FileTypeClassifier.Common;
using FileTypeClassifier.ML.Objects;
using System;
using System.IO;

namespace FileTypeClassifier.ML
{
    public class FeatureExtractor
    {
        private void ExtractFolder(string folderPath, string outputFile)
        {
            if (!Directory.Exists(folderPath))
            {
                Console.WriteLine($"{folderPath} does not exist");
                return;
            }

            var files = Directory.GetFiles(folderPath);
            using (var streamWriter = new StreamWriter(Path.Combine(AppContext.BaseDirectory, $"../../../Data/{outputFile}")))
            {
                foreach (var file in files)
                {
                    var extractData = new FileData(File.ReadAllBytes(file), file);
                    streamWriter.WriteLine(extractData.ToString());
                }
            }

            Console.WriteLine($"Extracted {files.Length} to {outputFile}");
        }

        public void Extract(string trainingPath, string testPath)
        {
            ExtractFolder(trainingPath, Constants.SAMPLE_DATA);
            ExtractFolder(testPath, Constants.TEST_DATA);
        }
    }
}