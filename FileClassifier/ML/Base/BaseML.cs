using FileClassifier.Common;
using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace FileClassifier.ML.Base
{
    public class BaseML
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, Constants.MODEL_FILENAME);

        protected readonly MLContext mlContext;
        private static Regex _stringRex;

        protected BaseML()
        {
            mlContext = new MLContext(2020);

            // This is critical to utilize the Windows-1252 encoding. Windows Executables use this encoding.
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            _stringRex = new Regex(@"[ -~\t]{8,}", RegexOptions.Compiled);
        }

        protected string GetStrings(byte[] data)
        {
            var stringLines = new StringBuilder();

            if (data == null || data.Length == 0)
            {
                return stringLines.ToString();
            }

            using (var ms = new MemoryStream(data, false))
            {
                using (var streamReader = new StreamReader(ms, Encoding.GetEncoding(1252), false, 2048, false))
                {
                    while (!streamReader.EndOfStream)
                    {
                        var line = streamReader.ReadLine();

                        if (string.IsNullOrEmpty(line))
                        {
                            continue;
                        }

                        line = line.Replace("^", "").Replace(")", "").Replace("-", "");

                        stringLines.Append(string.Join(string.Empty,
                            _stringRex.Matches(line)
                                      .Where(a => !string.IsNullOrEmpty(a.Value) &&
                                                  !string.IsNullOrWhiteSpace(a.Value))
                                      .ToList()));
                    }
                }
            }

            return string.Join(string.Empty, stringLines);
        }
    }
}