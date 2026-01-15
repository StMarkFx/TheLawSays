import { Feather } from "@expo/vector-icons";
import React, { useEffect, useMemo, useRef } from "react";
import {
  ActivityIndicator,
  Animated,
  Linking,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from "react-native";

import { colors, fonts } from "../theme";

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  loading?: boolean;
}

const LINKS = [
  {
    label: "GitHub: github.com/StMarkFx/TheLawSays",
    url: "https://github.com/StMarkFx/TheLawSays"
  },
  {
    label: "linkedin.com/in/stmarkadebayo",
    url: "https://linkedin.com/in/stmarkadebayo"
  },
  {
    label: "github.com/StMarkFx",
    url: "https://github.com/StMarkFx"
  },
  {
    label: "stmarkadebayo@gmail.com",
    url: "mailto:stmarkadebayo@gmail.com"
  }
];

export function Sidebar({ open, onClose, loading }: SidebarProps) {
  const translateX = useRef(new Animated.Value(-360)).current;

  useEffect(() => {
    Animated.timing(translateX, {
      toValue: open ? 0 : -360,
      duration: 220,
      useNativeDriver: true
    }).start();
  }, [open, translateX]);

  const overlayStyle = useMemo(
    () => [styles.overlay, { opacity: open ? 1 : 0 }],
    [open]
  );

  return (
    <>
      {open && (
        <Pressable style={overlayStyle} onPress={onClose} accessibilityLabel="Close sidebar" />
      )}
      <Animated.View style={[styles.container, { transform: [{ translateX }] }]}> 
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Project Info</Text>
          <TouchableOpacity onPress={onClose} accessibilityLabel="Close sidebar">
            <Feather name="x" size={20} color={colors.muted} />
          </TouchableOpacity>
        </View>

        <View style={styles.section}>
          <Text style={styles.bodyText}>
            TheLawSays is an open-source AI legal assistant built to democratize access to Nigerian law. It uses
            retrieval-augmented generation to deliver citation-backed answers from Federal and Lagos State statutes.
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>About St. Mark</Text>
          <Text style={styles.bodyText}>
            St. Mark Adebayo is an AI/ML Engineer who created TheLawSays, a Nigerian legal assistant designed to
            democratize access to legal information through advanced RAG technology.
          </Text>
          <Text style={styles.bodyText}>
            He focuses on applying AI to real-world challenges across justice, education, innovation, and community
            growth.
          </Text>
          <View style={styles.linkGroup}>
            {LINKS.map((item) => (
              <TouchableOpacity
                key={item.url}
                onPress={() => Linking.openURL(item.url)}
                accessibilityLabel={item.label}
              >
                <Text style={styles.link}>{item.label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator color={colors.accent} />
          </View>
        )}
      </Animated.View>
    </>
  );
}

const styles = StyleSheet.create({
  overlay: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0,0,0,0.7)",
    zIndex: 10
  },
  container: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    width: 320,
    maxWidth: "85%",
    backgroundColor: colors.card,
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 32,
    zIndex: 20
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border
  },
  headerTitle: {
    color: colors.muted,
    fontSize: 12,
    letterSpacing: 2,
    textTransform: "uppercase",
    fontFamily: fonts.medium
  },
  section: {
    paddingTop: 20
  },
  sectionTitle: {
    color: colors.muted,
    fontSize: 12,
    letterSpacing: 2,
    textTransform: "uppercase",
    fontFamily: fonts.medium
  },
  bodyText: {
    color: colors.foreground,
    fontSize: 14,
    lineHeight: 20,
    fontFamily: fonts.regular,
    marginBottom: 12
  },
  linkGroup: {
    marginTop: 4
  },
  link: {
    color: colors.foreground,
    fontSize: 14,
    textDecorationLine: "underline",
    textDecorationStyle: "dotted",
    fontFamily: fonts.regular,
    marginBottom: 10
  },
  loadingOverlay: {
    position: "absolute",
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(5,5,5,0.85)",
    alignItems: "center",
    justifyContent: "center"
  }
});
