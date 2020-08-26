using FileTypeClassifier.Enums;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Text;

namespace FileTypeClassifier.ML.Objects
{
    public class FileData
    {
        private const float TRUE = 1.0f;
        private const float FALSE = 0.0f;

        [ColumnName("Label")]
        public float Label { get; set; }

        public float isBinary { get; set; }
        public float isMZHeader { get; set; }
        public float isPKHeader { get; set; }

        private bool HasBinaryContent(Span<byte> fileContent) =>
            Encoding.UTF8.GetString(fileContent.ToArray()).Any(a => char.IsControl(a) && a != '\r' && a != '\n');

        private bool HasHeaderBytes(Span<byte> data, string match) =>
            Encoding.UTF8.GetString(data) == match;

        public override string ToString() => 
            $"{Label}, {isBinary}, {isMZHeader}, {isPKHeader}";

        public FileData(Span<byte> data, string fileName = null)
        {
            // Used for training purposes only
            if (!string.IsNullOrEmpty(fileName))
            {
                if (fileName.Contains("ps1"))
                {
                    Label = (float)FileTypes.Script;
                }
                else if (fileName.Contains("exe"))
                {
                    Label = (float)FileTypes.Executable;
                }
                else if (fileName.Contains("doc"))
                {
                    Label = (float)FileTypes.Document;
                }
            }

            isBinary = HasBinaryContent(data) ? TRUE : FALSE;

            isMZHeader = HasHeaderBytes(data.Slice(0, 2), "MZ") ? TRUE : FALSE;

            isPKHeader = HasHeaderBytes(data.Slice(0, 2), "PK") ? TRUE : FALSE;
        }

        /// <summary>
        /// Used for mapping cluster ids to results only
        /// </summary>
        /// <param name="fileType"></param>
        public FileData(FileTypes fileType)
        {
            Label = (float)fileType;

            switch (fileType)
            {
                case FileTypes.Executable:
                    isBinary = TRUE;
                    isMZHeader = TRUE;
                    isPKHeader = FALSE;
                    break;

                case FileTypes.Document:
                    isBinary = TRUE;
                    isMZHeader = FALSE;
                    isPKHeader = TRUE;
                    break;

                case FileTypes.Script:
                    isBinary = FALSE;
                    isMZHeader = FALSE;
                    isPKHeader = FALSE;
                    break;

                default:
                    break;
            }
        }
    }
}