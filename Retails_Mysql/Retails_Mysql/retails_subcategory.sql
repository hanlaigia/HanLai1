-- MySQL dump 10.13  Distrib 8.0.32, for Win64 (x86_64)
--
-- Host: localhost    Database: retails
-- ------------------------------------------------------
-- Server version	8.0.32

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `subcategory`
--

DROP TABLE IF EXISTS `subcategory`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `subcategory` (
  `SubcategoryID` int NOT NULL,
  `CategoryID` int DEFAULT NULL,
  `Name` text COLLATE utf8mb3_unicode_ci,
  PRIMARY KEY (`SubcategoryID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `subcategory`
--

LOCK TABLES `subcategory` WRITE;
/*!40000 ALTER TABLE `subcategory` DISABLE KEYS */;
INSERT INTO `subcategory` VALUES (1,1,'Mountain Bikes'),(2,1,'Road Bikes'),(3,1,'Touring Bikes'),(4,2,'Handlebars'),(5,2,'Bottom Brackets'),(6,2,'Brakes'),(7,2,'Chains'),(8,2,'Cranksets'),(9,2,'Derailleurs'),(10,2,'Forks'),(11,2,'Headsets'),(12,2,'Mountain Frames'),(13,2,'Pedals'),(14,2,'Road Frames'),(15,2,'Saddles'),(16,2,'Touring Frames'),(17,2,'Wheels'),(18,3,'Bib-Shorts'),(19,3,'Caps'),(20,3,'Gloves'),(21,3,'Jerseys'),(22,3,'Shorts'),(23,3,'Socks'),(24,3,'Tights'),(25,3,'Vests'),(26,4,'Bike Racks'),(27,4,'Bike Stands'),(28,4,'Bottles and Cages'),(29,4,'Cleaners'),(30,4,'Fenders'),(31,4,'Helmets'),(32,4,'Hydration Packs'),(33,4,'Lights'),(34,4,'Locks'),(35,4,'Panniers'),(36,4,'Pumps'),(37,4,'Tires and Tubes');
/*!40000 ALTER TABLE `subcategory` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-03-14 18:13:47
